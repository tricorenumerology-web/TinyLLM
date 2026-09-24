"""Character-level tokenizer.

The simplest possible tokenizer: every unique character in the training text
becomes a token. It has a tiny vocabulary, needs no external dependencies,
and is the classic choice for very small language models (cf. char-rnn,
nanoGPT's char mode).
"""

from __future__ import annotations

import json


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    # --------------------------------------------------------------- encode
    def encode(self, text: str) -> list[int]:
        out = []
        for ch in text:
            if ch not in self.stoi:
                raise KeyError(f"character {ch!r} not in vocabulary")
            out.append(self.stoi[ch])
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ----------------------------------------------------------- (de)serial
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model_type": "char", "chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("model_type") != "char":
            raise ValueError(f"not a char tokenizer file: {path}")
        return cls(obj["chars"])
