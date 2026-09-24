"""A tiny GPT-style language model written with pure NumPy.

No PyTorch, no autograd engine: every layer implements its own forward pass
and its own hand-derived backward pass (back-propagation). The only
dependency is NumPy.

Architecture (decoder-only transformer, GPT-2 style)::

    token embedding + learned positional embedding
      -> N x [ LayerNorm -> multi-head causal self-attention -> + residual
               LayerNorm -> MLP with GELU                    -> + residual ]
      -> final LayerNorm -> linear head (optionally tied to the embedding)
      -> softmax + cross-entropy

For every layer ``y = f(x)`` the code also implements ``dx = f'(x).dy``
plus the gradients of the layer's parameters, so a full backward pass is
just the composition of the per-layer backward functions, executed in
reverse order.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import ModelConfig

_GELU_C = float(np.sqrt(2.0 / np.pi))


# --------------------------------------------------------------------- math
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation, tanh approximation (as in GPT-2 / BERT)."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x ** 3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU w.r.t. ``x``."""
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _GELU_C * (1.0 + 3.0 * 0.044715 * x ** 2)


def softmax_lastdim(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------- layers
def layer_norm_forward(x, g, b, eps=1e-5):
    """LayerNorm over the last axis. Returns (output, backward_cache)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)                 # biased variance
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    out = g * xhat + b
    return out, (xhat, rstd, g)


def layer_norm_backward(dout, cache):
    """Backward through LayerNorm. Returns (dx, dg, db)."""
    xhat, rstd, g = cache
    dg = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
    db = dout.sum(axis=tuple(range(dout.ndim - 1)))
    dxhat = dout * g
    # dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    dx = rstd * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


def cross_entropy_forward(logits_flat: np.ndarray, targets_flat: np.ndarray):
    """Softmax cross-entropy on flat logits (N, V) and int targets (N,)."""
    n, _ = logits_flat.shape
    shifted = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1))
    logprobs = shifted - logsumexp[:, None]
    loss = float(-logprobs[np.arange(n), targets_flat].mean())
    cache = (logprobs, targets_flat, n, loss)
    return loss, cache


def cross_entropy_backward(cache):
    logprobs, targets_flat, n, _ = cache
    dlogits = np.exp(logprobs)                          # = softmax probs
    dlogits[np.arange(n), targets_flat] -= 1.0
    return dlogits / n


def linear_forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y = x @ w for 2-D or 3-D x; w has shape (in_features, out_features)."""
    return x @ w


def linear_backward(dout: np.ndarray, x: np.ndarray, w: np.ndarray):
    """Backward of y = x @ w. Returns (dx, dw)."""
    d2 = dout.reshape(-1, w.shape[1])
    x2 = x.reshape(-1, w.shape[0])
    dw = x2.T @ d2
    dx = (d2 @ w.T).reshape(x.shape)
    return dx, dw


# -------------------------------------------------------------------- model
class TinyLM:
    """Decoder-only transformer language model with manual backprop."""

    def __init__(self, cfg: ModelConfig, seed: int = 1337, dtype=np.float32):
        if cfg.vocab_size is None or cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be set (from the tokenizer)")
        cfg.head_size()  # validates n_embd / n_head
        self.cfg = cfg
        self.dtype = dtype
        self.training = True
        self.rng = np.random.default_rng(seed + 1)      # used only for dropout

        c, v, t = cfg.n_embd, cfg.vocab_size, cfg.block_size
        f = cfg.ffn_mult * c
        res_std = 0.02 / np.sqrt(2.0 * cfg.n_layer)     # GPT-2 residual init

        def norm(shape):
            return (np.ones(shape, dtype=dtype), np.zeros(shape, dtype=dtype))

        rng = np.random.default_rng(seed)
        p: dict[str, np.ndarray] = {}
        p["wte"] = (rng.standard_normal((v, c)) * 0.02).astype(dtype)
        p["wpe"] = (rng.standard_normal((t, c)) * 0.01).astype(dtype)
        for i in range(cfg.n_layer):
            h = f"h{i}."
            p[h + "ln1_g"], p[h + "ln1_b"] = norm((c,))
            p[h + "wq"] = (rng.standard_normal((c, c)) * 0.02).astype(dtype)
            p[h + "wk"] = (rng.standard_normal((c, c)) * 0.02).astype(dtype)
            p[h + "wv"] = (rng.standard_normal((c, c)) * 0.02).astype(dtype)
            p[h + "wo"] = (rng.standard_normal((c, c)) * res_std).astype(dtype)
            p[h + "ln2_g"], p[h + "ln2_b"] = norm((c,))
            p[h + "w_in"] = (rng.standard_normal((c, f)) * 0.02).astype(dtype)
            p[h + "w_out"] = (rng.standard_normal((f, c)) * res_std).astype(dtype)
        p["lnf_g"], p["lnf_b"] = norm((c,))
        if not cfg.tie_weights:
            p["lm_head"] = (rng.standard_normal((v, c)) * 0.02).astype(dtype)

        self.params = p
        self.grads = {k: np.zeros_like(w) for k, w in p.items()}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._cache: dict | None = None                 # filled by forward()

    # ------------------------------------------------------------ utilities
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(w.size for w in self.params.values())
        if non_embedding:
            n -= self.params["wte"].size + self.params["wpe"].size
        return n

    def train_mode(self) -> None:
        self.training = True

    def eval_mode(self) -> None:
        self.training = False

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0

    def _dropout(self, x: np.ndarray):
        """Inverted dropout. Returns (dropped_x, mask_or_None)."""
        if self.training and self.cfg.dropout > 0.0:
            keep = 1.0 - self.cfg.dropout
            mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
            return x * mask, mask
        return x, None

    def _causal_mask(self, t: int) -> np.ndarray:
        mask = self._mask_cache.get(t)
        if mask is None:
            mask = np.tril(np.ones((t, t), dtype=bool))
            self._mask_cache[t] = mask
        return mask

    # -------------------------------------------------------------- forward
    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """Forward pass.

        idx: (B, T) int array of token ids with T <= block_size.
        targets: optional (B, T) int array of next-token ids.

        Returns (logits, loss); loss is None when targets is None.
        Activations needed for :meth:`backward` are cached.
        """
        p, cfg = self.params, self.cfg
        b, t = idx.shape
        c = cfg.n_embd
        if t > cfg.block_size:
            raise ValueError(f"sequence length {t} > block_size {cfg.block_size}")
        idx = idx.astype(np.int64, copy=False)

        # --- token + positional embeddings: x = wte[idx] + wpe[:T]
        x = p["wte"][idx] + p["wpe"][:t]                            # (B, T, C)

        # --- transformer blocks
        block_caches = []
        for i in range(cfg.n_layer):
            x, cache = self._block_forward(i, x)
            block_caches.append(cache)

        # --- final layer norm + LM head
        xf, lnf_cache = layer_norm_forward(x, p["lnf_g"], p["lnf_b"])
        w_out = p["wte"] if cfg.tie_weights else p["lm_head"]
        logits = linear_forward(xf.reshape(-1, c), w_out.T).reshape(b, t, -1)

        self._cache = dict(emb_idx=idx, block_caches=block_caches,
                           lnf_cache=lnf_cache, xf=xf, t=t)

        loss = None
        if targets is not None:
            targets = targets.astype(np.int64, copy=False)
            loss, ce_cache = cross_entropy_forward(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            self._cache["ce_cache"] = ce_cache
        return logits, loss

    def _block_forward(self, i: int, x: np.ndarray):
        """Pre-norm residual block: x = x + mlp(ln2(x + attn(ln1(x))))."""
        p = self.params
        h = f"h{i}."

        # ---- attention sub-block
        xn1, ln1_cache = layer_norm_forward(x, p[h + "ln1_g"], p[h + "ln1_b"])
        attn_out, attn_cache = self._attention_forward(h, xn1)
        attn_out, drop1 = self._dropout(attn_out)
        x_mid = x + attn_out                                        # residual 1

        # ---- MLP sub-block
        xn2, ln2_cache = layer_norm_forward(x_mid, p[h + "ln2_g"], p[h + "ln2_b"])
        ffn_out, ffn_cache = self._ffn_forward(h, xn2)
        ffn_out, drop2 = self._dropout(ffn_out)
        x_out = x_mid + ffn_out                                     # residual 2

        cache = dict(h=h, x_in=x, ln1_cache=ln1_cache, attn_cache=attn_cache,
                     drop1=drop1, ln2_cache=ln2_cache, ffn_cache=ffn_cache,
                     drop2=drop2)
        return x_out, cache

    def _attention_forward(self, h: str, x: np.ndarray):
        """Multi-head causal self-attention over a (B, T, C) input."""
        p, cfg = self.params, self.cfg
        b, t, c = x.shape
        nh, hs = cfg.n_head, cfg.head_size()

        q = linear_forward(x, p[h + "wq"])                          # (B, T, C)
        k = linear_forward(x, p[h + "wk"])
        v = linear_forward(x, p[h + "wv"])
        # split heads: (B, T, C) -> (B, nh, T, hs)
        q = q.reshape(b, t, nh, hs).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, nh, hs).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, nh, hs).transpose(0, 2, 1, 3)

        scale = 1.0 / np.sqrt(hs)
        att = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale         # (B, nh, T, T)
        att = np.where(self._causal_mask(t), att, -np.inf)          # causal mask
        probs = softmax_lastdim(att)                                # rows sum to 1
        probs, drop_mask = self._dropout(probs)

        y = np.matmul(probs, v)                                     # (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(b, t, c)                # merge heads
        out = linear_forward(y, p[h + "wo"])
        cache = dict(h=h, x=x, q=q, k=k, v=v, probs=probs, y=y, scale=scale,
                     drop_mask=drop_mask)
        return out, cache

    def _ffn_forward(self, h: str, x: np.ndarray):
        """Position-wise MLP: GELU(x @ w_in) @ w_out."""
        p = self.params
        h_pre = linear_forward(x, p[h + "w_in"])
        hid = gelu(h_pre)
        hid, drop_mask = self._dropout(hid)
        out = linear_forward(hid, p[h + "w_out"])
        return out, dict(x=x, h_pre=h_pre, hid=hid, drop_mask=drop_mask)

    # ------------------------------------------------------------- backward
    def backward(self) -> float:
        """Backprop through the most recent forward pass (with targets).

        Fills ``self.grads`` (additively on top of ``zero_grad()``) and
        returns the loss of that forward pass.
        """
        assert self._cache is not None and "ce_cache" in self._cache, \
            "call forward(idx, targets) before backward()"
        g, p, cfg = self.grads, self.params, self.cfg
        c = cfg.n_embd
        b, t = self._cache["emb_idx"].shape
        n = b * t

        # ---- cross-entropy -> dlogits (B, T, V)
        dlogits = cross_entropy_backward(self._cache["ce_cache"]).reshape(b, t, -1)

        # ---- LM head: logits = xf @ w_out.T   (w_out: (C, V))
        w_out = p["wte"] if cfg.tie_weights else p["lm_head"]
        dxf = (dlogits.reshape(n, -1) @ w_out)                      # (N, C)
        if cfg.tie_weights:
            g["wte"] += dlogits.reshape(n, -1).T @ self._cache["xf"].reshape(n, c)
        else:
            g["lm_head"] += dlogits.reshape(n, -1).T @ self._cache["xf"].reshape(n, c)

        # ---- final layer norm
        dx, dg, db = layer_norm_backward(dxf.reshape(b, t, c), self._cache["lnf_cache"])
        g["lnf_g"] += dg
        g["lnf_b"] += db

        # ---- transformer blocks in reverse order
        for cache in reversed(self._cache["block_caches"]):
            dx = self._block_backward(cache, dx)

        # ---- embeddings
        np.add.at(g["wte"], self._cache["emb_idx"].reshape(-1), dx.reshape(n, c))
        g["wpe"][:t] += dx.sum(axis=0)

        return self._cache["ce_cache"][3]

    def _block_backward(self, cache: dict, dout: np.ndarray) -> np.ndarray:
        """Backward of x_out = x_mid + ffn(ln2(x_mid)), x_mid = x + attn(ln1(x))."""
        g, p = self.grads, self.params
        h = cache["h"]

        # ---- MLP branch (upstream grad is dout * dropout_mask2)
        dout_eff = dout if cache["drop2"] is None else dout * cache["drop2"]
        dxn2, dw_in, dw_out = self._ffn_backward(
            cache["ffn_cache"], dout_eff, p[h + "w_in"], p[h + "w_out"])
        g[h + "w_in"] += dw_in
        g[h + "w_out"] += dw_out

        # ---- LayerNorm 2: upstream grad into ln2's output is dxn2
        dx_mid_ln2, dg2, db2 = layer_norm_backward(dxn2, cache["ln2_cache"])
        g[h + "ln2_g"] += dg2
        g[h + "ln2_b"] += db2
        dx_mid = dout + dx_mid_ln2                   # total grad wrt x_mid

        # ---- attention branch (upstream grad is dx_mid * dropout_mask1)
        dxn1_eff = dx_mid if cache["drop1"] is None else dx_mid * cache["drop1"]
        dxn1, dwq, dwk, dwv, dwo = self._attention_backward(cache["attn_cache"], dxn1_eff)
        g[h + "wq"] += dwq
        g[h + "wk"] += dwk
        g[h + "wv"] += dwv
        g[h + "wo"] += dwo

        # ---- LayerNorm 1: upstream grad into ln1's output is dxn1
        dx_ln1, dg1, db1 = layer_norm_backward(dxn1, cache["ln1_cache"])
        g[h + "ln1_g"] += dg1
        g[h + "ln1_b"] += db1
        return dx_mid + dx_ln1                       # total grad wrt block input

    def _attention_backward(self, cache: dict, dout: np.ndarray):
        g, p, cfg = self.grads, self.params, self.cfg
        h = cache["h"]
        b, t, c = cache["x"].shape
        nh = cfg.n_head

        # ---- output projection: out = y @ wo
        dwo = cache["y"].reshape(-1, c).T @ dout.reshape(-1, c)
        dy = (dout @ p[h + "wo"].T).reshape(b, t, nh, -1).transpose(0, 2, 1, 3)

        # ---- y = probs @ v
        dprobs = np.matmul(dy, cache["v"].transpose(0, 1, 3, 2))
        dv = np.matmul(cache["probs"].transpose(0, 1, 3, 2), dy)
        if cache["drop_mask"] is not None:
            dprobs = dprobs * cache["drop_mask"]

        # ---- softmax backward (masked positions have probs == 0 -> zero grad)
        dscores = cache["probs"] * (
            dprobs - (dprobs * cache["probs"]).sum(axis=-1, keepdims=True))

        # ---- att = scale * (q @ k^T)
        dscores = dscores * cache["scale"]
        dq = np.matmul(dscores, cache["k"])                          # (B,nh,T,hs)
        dk = np.matmul(dscores.transpose(0, 1, 3, 2), cache["q"])

        # ---- merge heads and project back into the residual stream
        dq2 = dq.transpose(0, 2, 1, 3).reshape(b, t, c)
        dk2 = dk.transpose(0, 2, 1, 3).reshape(b, t, c)
        dv2 = dv.transpose(0, 2, 1, 3).reshape(b, t, c)
        dwq = cache["x"].reshape(-1, c).T @ dq2.reshape(-1, c)
        dwk = cache["x"].reshape(-1, c).T @ dk2.reshape(-1, c)
        dwv = cache["x"].reshape(-1, c).T @ dv2.reshape(-1, c)
        dxn1 = dq2 @ p[h + "wq"].T + dk2 @ p[h + "wk"].T + dv2 @ p[h + "wv"].T
        return dxn1, dwq, dwk, dwv, dwo

    def _ffn_backward(self, cache: dict, dout: np.ndarray, w_in: np.ndarray,
                      w_out: np.ndarray):
        dw_out = cache["hid"].reshape(-1, w_out.shape[0]).T @ dout.reshape(-1, w_out.shape[1])
        dhid = dout @ w_out.T
        if cache["drop_mask"] is not None:
            dhid = dhid * cache["drop_mask"]
        dh_pre = dhid * gelu_grad(cache["h_pre"])
        dw_in = cache["x"].reshape(-1, w_in.shape[0]).T @ dh_pre.reshape(-1, w_in.shape[1])
        dx = dh_pre @ w_in.T
        return dx, dw_in, dw_out

    # ------------------------------------------------------------ sampling
    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None,
                 rng: np.random.Generator | None = None) -> list[int]:
        """Autoregressive sampling (no KV cache: each step re-runs the context)."""
        rng = rng or np.random.default_rng()
        self.eval_mode()
        ids = list(prompt_ids) or [0]
        for _ in range(max_new_tokens):
            ctx = np.asarray(ids[-self.cfg.block_size:], dtype=np.int64)[None, :]
            logits, _ = self.forward(ctx)
            ids.append(sample_token(logits[0, -1], temperature, top_k, top_p, rng))
        return ids

    # ---------------------------------------------------------- persistence
    def save(self, out_dir: str, meta: dict | None = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "params.npz"),
                 **{k: v.astype(np.float32) for k, v in self.params.items()})
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.cfg.to_json())
        if meta:
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, out_dir: str, dtype=np.float32) -> "TinyLM":
        cfg = ModelConfig.from_json(os.path.join(out_dir, "config.json"))
        model = cls(cfg, seed=0, dtype=dtype)
        with np.load(os.path.join(out_dir, "params.npz")) as z:
            for k in model.params:
                if k not in z:
                    raise KeyError(f"checkpoint is missing parameter {k!r}")
                model.params[k][...] = z[k].astype(model.dtype)
        return model


def sample_token(logits: np.ndarray, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 rng: np.random.Generator | None = None) -> int:
    """Sample one token id from a logits vector (temperature / top-k / top-p)."""
    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float64)
    if temperature <= 0.0:                                   # greedy decoding
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.size:
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = softmax_lastdim(logits)
    if top_p is not None and 0.0 < top_p < 1.0:              # nucleus sampling
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        keep = (csum - probs[order]) < top_p                 # smallest prefix with sum >= p
        keep[0] = True
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()
    return int(rng.choice(probs.size, p=probs))
