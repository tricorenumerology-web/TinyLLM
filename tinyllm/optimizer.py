"""AdamW optimizer and gradient clipping, implemented with NumPy."""

from __future__ import annotations

import numpy as np


def clip_grad_global_norm(grads: dict[str, np.ndarray], max_norm: float) -> float:
    """Clip gradients in-place to a global L2 norm. Returns the pre-clip norm."""
    total = 0.0
    for g in grads.values():
        total += float(np.sum(g.astype(np.float64) ** 2))
    norm = float(np.sqrt(total))
    if max_norm is not None and max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads.values():
            g *= scale
    return norm


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Weight decay is applied only to weight matrices (``ndim >= 2``), not to
    LayerNorm gains/biases -- the usual convention.
    """

    def __init__(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray],
                 lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k, p in self.params.items():
            g = self.grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            if self.weight_decay > 0 and p.ndim >= 2:
                p *= (1.0 - self.lr * self.weight_decay)   # decoupled decay
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0
