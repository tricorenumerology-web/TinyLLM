"""Data loading and batching utilities."""

from __future__ import annotations

import numpy as np


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def to_ids(text: str, tokenizer) -> np.ndarray:
    ids = tokenizer.encode(text)
    dtype = np.int16 if tokenizer.vocab_size < 2 ** 15 else np.int32
    return np.asarray(ids, dtype=dtype)


def train_val_split(ids: np.ndarray, val_frac: float = 0.1):
    """Split a token stream into (train, val) without shuffling."""
    n = len(ids)
    n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
    return ids[:-n_val] if n_val else ids, ids[-n_val:] if n_val else ids[:0]


def get_batch(data: np.ndarray, batch_size: int, block_size: int,
              rng: np.random.Generator):
    """Sample a random (inputs, targets) batch from a 1-D token stream."""
    if len(data) <= block_size + 1:
        raise ValueError(
            f"split has {len(data)} tokens, needs at least block_size+1={block_size + 1}")
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix]).astype(np.int64)
    return x, y


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int,
              min_lr_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * base_lr``."""
    min_lr = base_lr * min_lr_frac
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * min(progress, 1.0)))
