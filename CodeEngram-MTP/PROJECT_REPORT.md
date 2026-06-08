# CodeEngram-MTP-120M Project Report

## 1. Project Goal

CodeEngram-MTP-120M is a small research-style coding language model prototype for Python DSA problems. The goal is not to compete with large coding LLMs. The goal is to demonstrate a clean ML-systems pipeline with three ideas:

1. **Teacher distillation** for structured reasoning/code/test data.
2. **Multi-Token Prediction (MTP)** for future-token learning.
3. **Engram memory injection** for algorithmic fact recall.

The project is designed for portfolio, interview discussion, and ablation-based experimentation.

## 2. Architecture

The final architecture is staged:

```text
Input tokens
↓
Token embeddings
↓
Decoder Transformer blocks with RoPE attention
↓
Optional Engram n-gram memory injection after a chosen block
↓
Final RMSNorm
↓
LM head predicts t+1
Optional MTP head predicts t+2
```

Main training objective:

```text
loss = lm_loss + 0.3 * mtp_loss + 0.01 * gate_regularization
```

## 3. Implemented Phases

| Phase | Status | Purpose |
|---|---:|---|
| Phase 0 | Done | Project skeleton and config |
| Phase 1A | Done | JSONL dataset loader |
| Phase 1B | Done | Tokenizer foundation |
| Phase 2A | Done | Base Transformer |
| Phase 2B | Done | SFT training loop |
| Phase 2C | Done | Checkpoint + inference |
| Phase 3A | Done | MTP architecture |
| Phase 3B | Done | MTP training |
| Phase 4A | Done | Engram memory table |
| Phase 4B | Done | Engram injection gate |
| Phase 5 | Done | Joint MTP + Engram training |
| Phase 6 | Done | Evaluation + ablation |
| Phase 7 | Done | Teacher distillation scaffold |
| Phase 8 | Done | GRPO scaffold |
| Phase 9 | Done | CLI demo + GitHub polish |

## 4. Evaluation Plan

The project supports ablation across:

1. Base Transformer
2. Base + MTP
3. Base + Engram
4. Base + MTP + Engram
5. Final + GRPO-style reward tuning

Planned real metrics after training:

- Loss / perplexity
- Python syntax validity
- Unit-test pass rate
- Complexity-analysis accuracy
- Engram factual recall
- MTP future-token accuracy

## 5. Data Plan

The current repo includes tiny mock data for pipeline checks. The real training plan is:

1. Start with open coding datasets such as MBPP, APPS, and CodeContests-style sources.
2. Filter for Python DSA tasks.
3. Use teacher models to generate structured reasoning, algorithms, code, complexity, and tests.
4. Keep only samples that pass syntax checks, unit tests, format validation, deduplication, and length constraints.

Target:

```text
First serious dataset: 5k–20k high-quality samples
Later dataset: 50k–100k high-quality distilled samples
```

## 6. Hardware Plan

Recommended first training setup:

```text
GPU: Kaggle/Colab T4 16GB
Precision: fp16/bf16
Context length: 512 first, then 1024
Micro-batch: 1–2
Gradient accumulation: 16–32
```

Do not begin long training until small sanity runs and evaluation scripts pass.

## 7. Resume Summary

**CodeEngram-MTP-120M** — Built a research-style Python DSA coding LLM prototype with a decoder Transformer backbone, Multi-Token Prediction head, hash-based Engram memory injection, teacher-distilled training data pipeline, GRPO-style reward tuning scaffold, and ablation-based evaluation.
