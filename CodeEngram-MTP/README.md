# CodeEngram-MTP-120M

A compact research-style **Python DSA coding LLM prototype** with:

- decoder-only Transformer backbone
- RoPE causal attention
- Multi-Token Prediction (MTP) head
- hash-based Engram memory injection gate
- supervised fine-tuning pipeline
- teacher distillation scaffold
- GRPO-style reasoning tuning scaffold
- evaluation + ablation scripts

This repository is designed as a portfolio/interview project. The goal is to prove the architecture and pipeline with clean engineering, not to claim state-of-the-art performance.

---

## Architecture

```text
Input tokens
↓
Token embeddings
↓
Transformer decoder blocks
↓
Optional Engram memory injection
↓
Final norm
↓
LM head predicts t+1
Optional MTP head predicts t+2
```

Joint objective:

```text
loss = lm_loss + 0.3 * mtp_loss + 0.01 * gate_regularization
```

---

## Project status

| Component | Status |
|---|---:|
| Project skeleton | Done |
| Dataset loader | Done |
| Tokenizer | Done |
| Base Transformer | Done |
| SFT training | Done |
| Checkpoint + inference | Done |
| MTP architecture | Done |
| MTP training | Done |
| Engram memory table | Done |
| Engram injection gate | Done |
| Joint training | Done |
| Evaluation + ablation | Done |
| Teacher distillation scaffold | Done |
| GRPO scaffold | Done |
| CLI demo + GitHub polish | Done |

---

## Folder structure

```text
CodeEngram-MTP/
├── configs/                 # model/training configuration
├── data/                    # raw, teacher, processed, and Engram fact data
├── src/
│   ├── tokenizer/           # tokenizer training/loading
│   ├── model/               # Transformer, attention, MTP, Engram, final model
│   ├── training/            # SFT, MTP, Engram, GRPO training loops
│   ├── distillation/        # teacher-output generation and merge scripts
│   ├── evaluation/          # unit tests, code eval, memory eval, ablation
│   └── utils/               # dataset, losses, generation, checkpointing
├── scripts/                 # runnable checks and demos
├── checkpoints/             # local checkpoints, ignored by git
├── PROJECT_REPORT.md
└── README.md
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run checks

Quick checks:

```bash
python scripts/run_all_checks.py --quick
```

Full checks, including tiny training loops:

```bash
python scripts/run_all_checks.py
```

Individual checks:

```bash
python scripts/check_project.py
python scripts/check_dataset.py
python scripts/check_tokenizer.py
python scripts/check_model.py
python scripts/check_mtp.py
python scripts/check_engram.py
python scripts/check_engram_injection.py
python scripts/check_joint_training.py
python scripts/check_evaluation.py
python scripts/check_distillation.py
python scripts/check_grpo.py
python scripts/check_phase9.py
```

---

## Run demo

Smoke test without checkpoint:

```bash
python scripts/demo.py --random-init --cpu --prompt "### Problem\nWrite binary search in Python." --max-new-tokens 20
```

After a tiny SFT check creates a checkpoint:

```bash
python scripts/check_training.py
python scripts/demo.py --checkpoint checkpoints/sft_tiny_check.pt --cpu --prompt "### Problem\nWrite a function to add two numbers."
```

Important: tiny checkpoints are not intelligent. They only prove the software path works.

---

## Training path

Correct training order:

```text
1. Tiny sanity run
2. Small SFT run
3. MTP run
4. Engram run
5. Joint MTP + Engram run
6. Evaluation + ablation
7. Small GRPO-style tuning
8. Larger Kaggle/Colab training only after checks pass
```

Recommended first real training setup:

```text
GPU: Kaggle/Colab T4 16GB
Precision: fp16/bf16
Context length: 512 first
Micro-batch: 1–2
Gradient accumulation: 16–32
Dataset: 5k–20k high-quality distilled samples first
```

---

## Data plan

The repository includes tiny mock examples for testing. Real training data should come from open programming datasets, then be distilled into this format:

```text
Problem
Reasoning
Algorithm
Code
Complexity
Tests
```

Quality filters:

- Python syntax parses successfully
- unit tests pass
- complexity section exists
- output follows exact format
- duplicates removed
- sample fits context length

Target dataset sizes:

```text
First serious run: 5k–20k samples
Later stronger run: 50k–100k samples
```

---

## Ablation goal

The project is designed to compare:

| Variant | Purpose |
|---|---|
| Base Transformer | baseline |
| Base + MTP | tests future-token prediction benefit |
| Base + Engram | tests memory benefit |
| Base + MTP + Engram | tests combined architecture |
| Final + GRPO | tests reward tuning benefit |

Final README results should include real numbers after training, such as pass@1, syntax validity, factual recall, and complexity accuracy.

---

## Resume line

**CodeEngram-MTP-120M** — Built a research-style Python DSA coding LLM prototype with a decoder Transformer backbone, Multi-Token Prediction head, hash-based Engram memory injection, teacher-distilled training data pipeline, GRPO-style reward tuning scaffold, and ablation-based evaluation.

GitHub link format:

```text
https://github.com/YOUR_USERNAME/CodeEngram-MTP-120M
```

---

## Current limitation

This repository currently proves the full software pipeline using tiny data and tiny CPU-safe checks. It does not yet contain a fully trained high-quality coding model. Real performance requires distilled data and GPU training.
