# 🧠 Efficient Multilingual SLM

> A compact, from-scratch GPT-style language model (1.83M parameters) trained on a mixed **English + Hindi** corpus under extreme resource constraints — **CPU-only, 2,000 optimizer steps, ≤ 2M parameters**.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)


---

## ✨ Highlights

| Metric | Value |
|---|---|
| **Final Dev BPB** | **1.7067** |
| **Improvement over baseline** | **28.0% relative** (from 2.3718 → 1.7067) |
| **Total Parameters** | 1,827,968 / 2,000,000 cap |
| **Training Steps** | 2,000 (hard cap) |
| **Hardware** | CPU only — no GPU required |

---

## 🏗️ Architecture

This is not a toy model — every architectural choice targets the specific constraints of this project:

- **Tokenizer** — SentencePiece Unigram (vocab = 8,192) with byte-fallback for lossless UTF-8 coverage, including Devanagari script. Default SentencePiece settings silently break Hindi text; our `train_unigram.py` fixes this.
- **Positional Encoding** — [RoPE](https://arxiv.org/abs/2104.09864) (Rotary Position Embeddings) replaces learned absolute positions for better length generalization.
- **Normalization** — RMSNorm instead of LayerNorm, reducing per-layer overhead.
- **Feed-Forward** — [SwiGLU](https://arxiv.org/abs/2002.05202) MLP (hidden dim scaled to ~2/3 to maintain parameter parity with a standard 4× GELU MLP).
- **Initialization** — GPT-2-style scaled residual init on each block's output projection to prevent variance blow-up with depth.
- **Weight Tying** — Token embedding and output head share weights, saving ~1M parameters.

### Optimizer: Why Plain Adam Beat Muon

We implemented the [Muon optimizer](https://arxiv.org/abs/2402.16494) (Newton-Schulz orthogonalized updates) from scratch and ran a controlled ablation:

| Optimizer | Dev BPB | Wall Time |
|---|---|---|
| Muon + AdamW (split) | 1.7210 | 669s |
| **Plain Adam (final)** | **1.7067** | 696s |

Muon lost by **0.83% relative BPB**. We shipped the simpler, better-scoring Adam run. The full analysis is in [`RUNLOG.md`](starter/RUNLOG.md) — this is a documented negative result, not a hidden one.

---

## 📊 Results Summary

| Configuration | Dev BPB | Parameters |
|---|---|---|
| Byte tokenizer + Adam (baseline) | 2.3718 | 1,339,840 |
| + Unigram tokenizer swap only | 1.9326 | 1,829,520 |
| + RoPE / RMSNorm / SwiGLU + Muon | 1.7210 | 1,827,968 |
| **+ Ablated Muon → Plain Adam (final)** | **1.7067** | **1,827,968** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PyTorch 2.0+
- `sentencepiece`

```bash
pip install torch sentencepiece
```

### Training Pipeline

```bash
cd starter

# Step 1 — Train the tokenizer
python train_unigram.py --input ../data/train_corpus.txt --vocab_size 8192

# Step 2 — Train the model (2,000 steps, ~11 min on CPU)
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt

# Step 3 — Evaluate
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

### Optimizer Ablation

To reproduce the Muon vs. Adam comparison:

```bash
# Run with Muon + AdamW (default)
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt_muon.pt

# Run with plain Adam only
python train.py --data ../data/train_corpus.txt --steps 2000 --optimizer adam_all --out ckpt_adam_only.pt
```

> **Note:** Data files are not included in this repository. Supply your own mixed-language text corpus as `data/train_corpus.txt` and `data/dev_eval.txt`.

---

## 📁 Project Structure

```
sLLM/
├── README.md               ← You are here
├── .gitignore
└── starter/
    ├── model.py             # GPT architecture (RoPE, RMSNorm, SwiGLU)
    ├── train.py             # Training loop with Muon/AdamW and LR scheduling
    ├── evaluate.py          # BPB evaluation script
    ├── muon.py              # From-scratch Muon optimizer implementation
    ├── tokenizer.py         # Tokenizer loading utilities
    ├── train_unigram.py     # SentencePiece Unigram tokenizer training
    ├── ckpt.pt              # Final checkpoint (Adam-only, BPB = 1.7067)
    ├── ckpt_adam_only.pt    # Ablation checkpoint
    ├── tok_v8192.model      # Trained tokenizer model
    ├── tok_v8192.vocab      # Tokenizer vocabulary
    ├── RUNLOG.md            # Complete experiment log with all runs
    ├── NOTES.md             # Design decisions and rationale
    └── SUMMARY.html         # Visual summary report
```

---

## 📝 Key Design Decisions

1. **Tokenizer fixes for Hindi** — Default SentencePiece uses NFKC normalization and a dummy prefix that silently corrupt Devanagari. We disable both and use identity normalization with byte-fallback, ensuring lossless round-tripping.

2. **SwiGLU hidden-dim scaling** — SwiGLU uses 3 matrices (gate, up, down) vs. GELU's 2. We shrink hidden dim by ~2/3 to maintain roughly the same parameter count, following standard practice.

3. **Negative results documented** — The Muon optimizer underperformed Adam in our regime. Rather than hiding this, we document it with analysis in the RUNLOG — the likely causes are un-tuned per-optimizer LR and the small batch size (8) dampening Muon's advantage.

---


<p align="center">
  <i>Built from scratch with PyTorch — no pretrained weights, no GPU, no shortcuts.</i>
</p>