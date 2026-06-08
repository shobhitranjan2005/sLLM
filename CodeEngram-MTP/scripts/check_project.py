from pathlib import Path
import yaml


REQUIRED_PATHS = [
    "configs/model_120m.yaml",

    "data/raw",
    "data/teacher_outputs",
    "data/processed",
    "data/engram_facts",

    "src/tokenizer",
    "src/model/transformer.py",
    "src/model/attention.py",
    "src/model/mtp.py",
    "src/model/engram.py",
    "src/model/codeengram_model.py",

    "src/training/train_sft.py",
    "src/training/train_mtp.py",
    "src/training/train_engram.py",
    "src/training/train_grpo.py",

    "src/distillation/generate_reasoning.py",
    "src/distillation/generate_code.py",
    "src/distillation/merge_teacher_outputs.py",

    "src/evaluation/run_unit_tests.py",
    "src/evaluation/eval_code.py",
    "src/evaluation/eval_memory.py",
    "src/evaluation/ablation.py",

    "src/utils/dataset.py",
    "src/utils/losses.py",
    "src/utils/generation.py",

    "checkpoints",
    "scripts",
    "README.md",
    "requirements.txt",
]


def check_paths():
    missing = []

    for path in REQUIRED_PATHS:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print("Missing paths:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("All required project paths exist.")


def check_config():
    config_path = Path("configs/model_120m.yaml")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"]["hidden_size"] == 768
    assert config["model"]["num_layers"] == 12
    assert config["model"]["num_attention_heads"] == 12
    assert config["mtp"]["enabled"] is False
    assert config["engram"]["enabled"] is False

    print("Config loaded successfully.")
    print(f"Model name: {config['model']['name']}")
    print(f"Hidden size: {config['model']['hidden_size']}")
    print(f"Layers: {config['model']['num_layers']}")
    print(f"Heads: {config['model']['num_attention_heads']}")


if __name__ == "__main__":
    check_paths()
    check_config()
    print("Phase 0 setup is complete.")
