"""Tokenizer interface used by train.py / evaluate.py / generate.py.

Two interface rules from the assignment brief (do not break these):
  1. tokenizer.load() must encode arbitrary UTF-8 text (keep a byte fallback).
  2. decode(encode(text)) == text for ANY UTF-8 text, not just text that
     resembles the training corpus. evaluate.py enforces this and raises
     SystemExit if it's violated.

This wraps a SentencePiece Unigram model (trained by train_unigram.py on
train_corpus.txt only, byte_fallback=True) and falls back to a pure
byte-level tokenizer if no trained model file is found.
"""
import glob
import os

import sentencepiece as spm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class SPTokenizer:
    """Lossless Unigram tokenizer with byte fallback."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.model_path = model_path

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(list(ids))


class ByteTokenizer:
    """Fallback: raw UTF-8 bytes, vocab_size=256. Trivially lossless."""

    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="strict")


def load(model_path=None):
    """Loads the trained Unigram tokenizer if present, else falls back to
    raw bytes. Set TOKENIZER_MODEL env var or pass model_path explicitly
    to pin a specific vocab (useful for the vocab-size sweep)."""
    if model_path is None:
        model_path = os.environ.get("TOKENIZER_MODEL")
    if model_path is None:
        candidates = sorted(glob.glob(os.path.join(_THIS_DIR, "tok_v*.model")))
        if candidates:
            model_path = max(candidates, key=os.path.getmtime)
    if model_path is None:
        print("[tokenizer] no tok_v*.model found -> falling back to byte "
              "tokenizer (vocab_size=256). Run train_unigram.py first.")
        return ByteTokenizer()
    return SPTokenizer(model_path)