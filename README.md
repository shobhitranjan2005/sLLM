# Efficient Multilingual SLM

A small (1.83M param) GPT-style language model trained from scratch on a
mixed English + Hindi corpus, under a hard 2,000-optimizer-step /
2,000,000-parameter / CPU-only budget.

**Final dev bits-per-byte: 1.7067** — a 28% relative improvement over an
unmodified byte-tokenizer baseline (2.3718 bpb) under the same caps.

## What's here
- A SentencePiece Unigram tokenizer with byte-fallback (lossless on
  arbitrary UTF-8, including Devanagari — default SentencePiece settings
  silently break this; see `starter/train_unigram.py` for the fix)
- A GPT variant using RoPE, RMSNorm, SwiGLU, and GPT-2-style scaled
  residual init instead of the more standard learned-position/LayerNorm/
  GELU baseline
- A from-scratch PyTorch implementation of the Muon optimizer
  (Newton-Schulz orthogonalized updates), benchmarked head-to-head
  against plain Adam
- A full experiment log (`starter/RUNLOG.md`) including a negative
  result: Muon underperformed plain Adam by ~0.8% relative bpb in this
  small-batch, short-step regime — documented and explained rather than
  hidden

## Results

| Config | Dev BPB | Params |
|---|---|---|
| Byte tokenizer, Adam (baseline) | 2.3718 | 1,339,840 |
| + Unigram tokenizer only | 1.9326 | 1,829,520 |
| + RoPE/RMSNorm/SwiGLU + Muon/AdamW | 1.7210 | 1,827,968 |
| **+ ablated Muon → plain Adam (final)** | **1.7067** | 1,827,968 |

Full reasoning for every change and every ablation is in
`starter/RUNLOG.md` and `starter/NOTES.md`.

## Run it yourself

Requires Python 3.12, PyTorch, and `sentencepiece`. Data files are not
included in this repo — supply your own mixed-language text corpus as
`data/train_corpus.txt` / `data/dev_eval.txt`.

```bash
cd starter
python train_unigram.py --input ../data/train_corpus.txt --vocab_size 8192
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

See `starter/HOW_TO_RUN.md` for the full workflow including the
optimizer ablation and vocab-size sweep.