"""Small GRPO-style reasoning/code tuning loop.

Phase 8 adds a CPU-safe prototype of reward-based tuning. It is intentionally
small and conservative: generate several candidate continuations, score them
with automatic rewards, normalize rewards inside the group, and update the
model toward higher-scoring candidates.

This is not a production RLHF system. It is the project scaffold we need before
running real unit-test-based GRPO on larger distilled data.
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.checkpointing import save_training_checkpoint
from src.utils.dataset import load_jsonl
from src.utils.generation import generate_token_ids


CONFIG_PATH = ROOT / "configs" / "model_120m.yaml"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "grpo_tiny.pt"


def load_yaml_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_config(raw_config: dict[str, Any], tokenizer_size: int, tiny: bool) -> CodeEngramConfig:
    """Build the final architecture config used for Phase 8 tuning."""
    config = CodeEngramConfig.from_yaml_dict(raw_config)
    config.vocab_size = tokenizer_size
    config.mtp_enabled = True
    config.mtp_num_heads = 1
    config.engram_enabled = True

    if tiny:
        config.max_seq_len = 64
        config.hidden_size = 64
        config.num_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 128
        config.dropout = 0.0
        config.engram_memory_slots = 128
        config.engram_memory_dim = 64
        config.engram_ngram_size = 3
        config.engram_injection_layer = 2
        config.engram_gate_regularization_weight = 0.01

    if config.engram_memory_dim != config.hidden_size:
        config.engram_memory_dim = config.hidden_size

    return config


def build_problem_prompt(problem: str) -> str:
    """Prompt used for reward tuning."""
    return (
        "### Problem\n"
        f"{problem}\n\n"
        "### Reasoning\n"
    )


def extract_code_block(text: str) -> str:
    """Extract likely Python code from model text.

    The tiny tokenizer does not preserve exact formatting well, so this function
    is deliberately permissive. For the real tokenizer, this can be stricter.
    """
    if "### Code" in text:
        text = text.split("### Code", 1)[1]
    if "### Complexity" in text:
        text = text.split("### Complexity", 1)[0]
    return text.strip()


def reward_candidate(text: str) -> float:
    """Automatic reward for one generated candidate.

    Reward components mirror the final plan, but are lightweight enough for a
    local check:
      + syntax/code-structure hints
      + complexity/reporting hints
      - contradiction/fake-output penalties
      - too-short/too-long penalties
    """
    reward = 0.0
    lowered = text.lower()

    if "### code" in lowered or "def " in text:
        reward += 0.4
    if "return" in text:
        reward += 0.2
    if "### complexity" in lowered or "time:" in lowered or "space:" in lowered:
        reward += 0.3
    if any(word in lowered for word in ["algorithm", "edge", "case", "because"]):
        reward += 0.2

    code = extract_code_block(text)
    if "def " in code:
        try:
            ast.parse(code)
            reward += 0.5
        except SyntaxError:
            reward -= 0.5

    if len(text.strip()) < 20:
        reward -= 0.3
    if len(text) > 1500:
        reward -= 0.2
    if any(bad in lowered for bad in ["i cannot", "as an ai", "fake", "unknown"]):
        reward -= 0.4

    return float(reward)


def sequence_logprob(
    model: CodeEngramForCausalLM,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Mean log probability of generated tokens after the prompt."""
    if full_ids.ndim != 2 or full_ids.size(0) != 1:
        raise ValueError("full_ids must have shape [1, seq_len].")
    if full_ids.size(1) <= prompt_len:
        raise ValueError("sequence must contain generated tokens after prompt.")

    outputs = model(input_ids=full_ids)
    logits = outputs["logits"][:, :-1, :]
    target_ids = full_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # Token at position i in target_ids is predicted from original position i.
    # Generated target tokens begin at index prompt_len - 1 in target_ids.
    generated_log_probs = token_log_probs[:, prompt_len - 1 :]
    return generated_log_probs.mean()


def grpo_step(
    model: CodeEngramForCausalLM,
    tokenizer: SimpleCodeTokenizer,
    problem: str,
    optimizer: AdamW,
    device: torch.device,
    group_size: int = 4,
    max_new_tokens: int = 24,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Run one tiny GRPO-style update for a single problem."""
    prompt = build_problem_prompt(problem)
    prompt_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long, device=device)
    prompt_len = prompt_ids.size(1)

    candidates: list[torch.Tensor] = []
    rewards: list[float] = []

    # Generate with no gradients; then recompute logprobs with gradients.
    model.eval()
    with torch.no_grad():
        for _ in range(group_size):
            full_ids = generate_token_ids(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=30,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(full_ids[0].tolist(), skip_special_tokens=True)
            candidates.append(full_ids.detach())
            rewards.append(reward_candidate(text))

    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    if reward_tensor.numel() > 1 and float(reward_tensor.std(unbiased=False).item()) > 1e-6:
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std(unbiased=False) + 1e-6)
    else:
        advantages = reward_tensor - reward_tensor.mean()

    model.train()
    losses: list[torch.Tensor] = []
    for full_ids, advantage in zip(candidates, advantages):
        full_ids = full_ids.to(device)
        logprob = sequence_logprob(model, full_ids=full_ids, prompt_len=prompt_len)
        losses.append(-advantage.detach() * logprob)

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "mean_reward": float(reward_tensor.mean().item()),
        "max_reward": float(reward_tensor.max().item()),
        "min_reward": float(reward_tensor.min().item()),
        "reward_spread": float((reward_tensor.max() - reward_tensor.min()).item()),
    }


def train_one_grpo_run(
    train_path: Path,
    checkpoint_path: Path,
    tiny: bool = True,
    max_steps: int = 3,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    seed: int = 7,
    max_new_tokens: int | None = None,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)

    raw_config = load_yaml_config()
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    config = build_grpo_config(raw_config, tokenizer_size=len(tokenizer), tiny=tiny)

    samples = load_jsonl(train_path)
    if not samples:
        raise ValueError("GRPO dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() and not tiny else "cpu")
    model = CodeEngramForCausalLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    last_stats: dict[str, float] = {}
    for step in range(1, max_steps + 1):
        sample = samples[(step - 1) % len(samples)]
        last_stats = grpo_step(
            model=model,
            tokenizer=tokenizer,
            problem=sample.problem,
            optimizer=optimizer,
            device=device,
            group_size=group_size,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else (20 if tiny else 64),
            temperature=1.0,
        )
        print(
            f"grpo step {step:04d} | loss {last_stats['loss']:.4f} | "
            f"reward mean {last_stats['mean_reward']:.4f} | "
            f"max {last_stats['max_reward']:.4f} | spread {last_stats['reward_spread']:.4f}"
        )

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        step=max_steps,
        extra={
            "phase": "8-GRPO",
            "group_size": group_size,
            "last_grpo_loss": last_stats.get("loss"),
            "last_mean_reward": last_stats.get("mean_reward"),
            "last_max_reward": last_stats.get("max_reward"),
            "last_reward_spread": last_stats.get("reward_spread"),
        },
    )

    last_stats["steps"] = float(max_steps)
    last_stats["parameters"] = float(model.count_parameters())
    return last_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use full config on GPU. Default is tiny CPU-safe mode.")
    args = parser.parse_args()

    stats = train_one_grpo_run(
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        tiny=not args.full,
        max_steps=args.max_steps,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
    )

    print("GRPO run complete.")
    print(f"Last GRPO loss: {stats['loss']:.4f}")
    print(f"Last mean reward: {stats['mean_reward']:.4f}")
    print(f"Last reward spread: {stats['reward_spread']:.4f}")


if __name__ == "__main__":
    main()
