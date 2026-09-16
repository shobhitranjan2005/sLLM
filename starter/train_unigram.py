"""Train a SentencePiece Unigram tokenizer on train_corpus.txt ONLY.

Why these specific SentencePiece flags (each guards losslessness or
Hindi/English fairness):

  byte_fallback=True        any byte not covered by a learned piece falls
                             back to a <0xXX> byte token instead of <unk>.
                             This is what makes the tokenizer lossless for
                             ARBITRARY UTF-8 text.
  normalization_rule_name   default SentencePiece applies NFKC unicode
    = "identity"             normalization, which silently rewrites some
                             Devanagari sequences (composed vs. decomposed
                             matras) and breaks exact round-trip.
                             "identity" disables all normalization.
  add_dummy_prefix=False    default SentencePiece prepends a virtual
                             leading space to every encoded string. That
                             space doesn't exist in the original text, so
                             decode(encode(text)) would gain a leading
                             space vs. the input.
  remove_extra_whitespaces  default SentencePiece collapses runs of
    = False                  whitespace, breaking losslessness if the
                             corpus/hidden file has meaningful runs.
  allow_whitespace_only_pieces=True
                             lets the model represent standalone
                             whitespace exactly.

character_coverage < 1.0 is safe here because byte_fallback=True means
rare glyphs just decompose into bytes instead of breaking anything —
it just means fewer pieces get "wasted" on one-off symbols.

Usage:
    python train_unigram.py --input ../data/train_corpus.txt --vocab_size 8192
"""
import argparse

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/train_corpus.txt")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--model_prefix", default=None,
                     help="default: tok_v{vocab_size}")
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    args = ap.parse_args()

    prefix = args.model_prefix or f"tok_v{args.vocab_size}"

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=args.character_coverage,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )
    print(f"wrote {prefix}.model and {prefix}.vocab")

    # immediate self-check: lossless round trip on the training text itself
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
    text = open(args.input, encoding="utf-8").read()
    ids = sp.encode(text, out_type=int)
    back = sp.decode(ids)
    ok = (back == text)
    print(f"vocab_size={sp.get_piece_size()}  tokens={len(ids):,}  "
          f"round_trip_ok={ok}")
    if not ok:
        for i, (a, b) in enumerate(zip(text, back)):
            if a != b:
                print(f"first mismatch at char {i}: {a!r} vs {b!r}")
                break
        raise SystemExit(
            "Tokenizer is NOT lossless on the training corpus itself. "
            "Do not proceed to training with this tokenizer.")


if __name__ == "__main__":
    main()