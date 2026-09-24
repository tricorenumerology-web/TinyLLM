"""Model and training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """Architecture hyper-parameters of the transformer."""

    vocab_size: int = 256        # set by the tokenizer when training starts
    block_size: int = 128        # maximum context length (in tokens)
    n_layer: int = 4             # number of transformer blocks
    n_head: int = 4              # number of attention heads
    n_embd: int = 128            # embedding / residual-stream width
    ffn_mult: int = 4            # MLP hidden width = ffn_mult * n_embd
    dropout: float = 0.0         # dropout probability (0 disables it)
    tie_weights: bool = True     # share token-embedding and output-head weights

    def head_size(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        return self.n_embd // self.n_head

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
