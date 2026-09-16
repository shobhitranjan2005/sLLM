# RUNLOG

## Run 0 — baseline (reference, not independently re-run in this session)
- Source: figures reported in the project's pre-existing README, produced
  with the unmodified starter code on the same train_corpus.txt/dev_eval.txt
  (assignment states data/steps/params are identical for every candidate).
- Byte tokenizer + Adam, baseline config: bpb = 2.3718, 1,339,840 params.
- Byte tokenizer + Unigram tokenizer swap only (no other changes): bpb = 1.9326.
- We did not re-run these ourselves after overwriting train.py/model.py;
  cited here only as a reference point for the deltas below.

## Run 1 — Unigram tokenizer (vocab 8192)
- Hypothesis: Devanagari costs ~3x a byte tokenizer's sequence length for
  no semantic gain; a subword tokenizer should raise effective context
  per training step.
- Command: `python train_unigram.py --input ../data/train_corpus.txt --vocab_size 8192`
- Result: round_trip_ok=True (verified lossless before training).
  1,582,803 tokens from 7,318,592 bytes (~4.6 bytes/token).
- Conclusion: lossless and ready to train against; no bpb number from this
  step alone (tokenizer-only, not a trained checkpoint).

## Run 2 — RoPE + RMSNorm + SwiGLU + scaled residual init + Muon/AdamW + warmup/cosine LR
- Hypothesis: combined, these should beat the reference Unigram baseline
  (1.9326 bpb) via more useful capacity per parameter and faster per-step
  convergence within the fixed 2,000-step budget.
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`
- Params: 1,827,968 / 2,000,000 cap. Muon: 778,240 params, AdamW: 1,049,728 params.
- Dev bpb: 1.7210
- Conclusion: strong improvement over the reference numbers, but this run
  confounds three changes at once (tokenizer + architecture + optimizer),
  so it doesn't isolate which change did the work on its own — see Run 3.

## Run 3 — ablation: Muon+AdamW vs. plain Adam, architecture held fixed
- Hypothesis: Muon's orthogonalized updates converge faster per-step than
  flat Adam on hidden-layer matrices, even in this small-batch (8),
  short-step (2,000) regime.
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --optimizer adam_all --out ckpt_adam_only.pt`
- Dev bpb: Muon+AdamW = 1.7210, Adam-only = 1.7067
- Wall time: 669s (Muon+AdamW) vs 696s (Adam-only) — essentially equal;
  we had initially assumed Muon would cost ~2x per step, which this
  measurement disproved.
- Conclusion: Muon LOST by 0.83% relative bpb, consistently across the
  entire training curve (Adam-only's logged loss was lower at every
  checkpoint from step 100 onward, not just the final eval). Most likely
  cause: this ablation reused the AdamW-tuned LR (3e-3) for the all-Adam
  run rather than independently sweeping Adam's LR, and Muon's LR (0.02)
  was not tuned either — this compares one arbitrarily-chosen LR per
  optimizer, not a fully controlled sweep. Batch size 8 (noisy per-step
  gradients) and the short 2,000-step budget likely also blunt Muon's
  usual advantage, which is more established at larger batch sizes and
  longer training runs. We are shipping the plain-Adam checkpoint as
  final: it is simpler (one optimizer, not two) AND measurably better
  here, so there's no tradeoff to justify keeping Muon. We did not spend
  further full 2,000-step runs (~11 min each) chasing a fix, given the
  marginal potential upside.

## Final submitted checkpoint
- `ckpt.pt` = the Run 3 "Adam-only" checkpoint (renamed/copied), NOT the
  Run 2 Muon checkpoint.
- Command used to produce it: `python train.py --data ../data/train_corpus.txt --steps 2000 --optimizer adam_all --out ckpt_adam_only.pt` (then copied to ckpt.pt)
- Final verified score: `{"bpb": 1.7067, "n_params": 1827968, "steps": 2000, "tokens_in_eval": 37273, "tokens_scored": 37272}`