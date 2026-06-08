"""Simple code-aware tokenizer for early pipeline testing.

This is not the final production tokenizer. It is a small, dependency-free
word/punctuation tokenizer so the project can test text -> ids -> text before
we train a real BPE/SentencePiece tokenizer on larger distilled data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z_][A-Za-z_0-9]*      | # identifiers/words
    \d+\.\d+|\d+               | # numbers
    ==|!=|<=|>=|->|\+=|-=|\*=|/=|//=|%=|\*\*|&&|\|\| | # multi-char operators
    \S                             # any remaining non-space char
    """,
    re.VERBOSE,
)


class SimpleCodeTokenizer:
    """Tiny tokenizer with a saved vocabulary.

    It supports the minimum operations we need before model building:
    tokenization, encoding, decoding, saving, and loading.
    """

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        for token in SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"Missing required special token: {token}")

        self.pad_token_id = self.token_to_id["<pad>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in self.tokenize(text)]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        special = set(SPECIAL_TOKENS)

        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)

        # Simple readable reconstruction. It is not exact byte-level detokenization.
        text = " ".join(tokens)
        for punct in [" .", " ,", " :", " ;", " )", " ]", " }"]:
            text = text.replace(punct, punct.strip())
        for punct in ["( ", "[ ", "{ "]:
            text = text.replace(punct, punct.strip())
        text = text.replace(" \n ", "\n")
        return text

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "SimpleCodeTokenizer",
            "special_tokens": SPECIAL_TOKENS,
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleCodeTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "SimpleCodeTokenizer":
            raise ValueError("Unsupported tokenizer file type")
        return cls(payload["token_to_id"])

    @classmethod
    def train_from_texts(cls, texts: Iterable[str], vocab_size: int = 512) -> "SimpleCodeTokenizer":
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must fit all special tokens")

        counts: Dict[str, int] = {}
        for text in texts:
            for token in cls.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        keep = [token for token, _ in sorted_tokens[: vocab_size - len(SPECIAL_TOKENS)]]

        token_to_id: Dict[str, int] = {}
        for token in SPECIAL_TOKENS + keep:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)
