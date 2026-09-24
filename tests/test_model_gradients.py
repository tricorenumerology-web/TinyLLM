"""Numerical gradient checks for the hand-written backprop.

Every parameter of a tiny model is perturbed by +/- eps and the resulting
change in loss (central finite differences) is compared against the
analytic gradient computed by ``model.backward()``. This is the strongest
possible validation that the hand-derived backward passes are correct.
"""

import numpy as np
import pytest

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM


def numerical_grad(model, name, idx, targets, eps=1e-5):
    """Central-difference gradient of the loss w.r.t. one parameter tensor."""
    p = model.params[name]
    g_num = np.zeros_like(p)
    flat = p.reshape(-1)
    gflat = g_num.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        _, lp = model.forward(idx, targets)
        flat[i] = orig - eps
        _, lm = model.forward(idx, targets)
        flat[i] = orig
        gflat[i] = (lp - lm) / (2 * eps)
    return g_num


def make_model(tie_weights, dropout=0.0):
    cfg = ModelConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2,
                      n_embd=16, ffn_mult=2, dropout=dropout,
                      tie_weights=tie_weights)
    return TinyLM(cfg, seed=42, dtype=np.float64)


@pytest.mark.parametrize("tie_weights", [True, False])
def test_full_model_gradient_check(tie_weights):
    model = make_model(tie_weights)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, model.cfg.vocab_size, size=(2, 8))
    targets = rng.integers(0, model.cfg.vocab_size, size=(2, 8))

    model.zero_grad()
    _, loss = model.forward(idx, targets)
    model.backward()

    total_checked = 0
    for name, p in model.params.items():
        g_num = numerical_grad(model, name, idx, targets)
        g_ana = model.grads[name]
        denom = max(1e-8, np.abs(g_num).sum() + np.abs(g_ana).sum())
        rel_err = np.abs(g_num - g_ana).sum() / denom
        assert rel_err < 1e-6, f"gradient mismatch for {name}: rel_err={rel_err:.3e}"
        total_checked += p.size
    # sanity: we really checked the whole parameter set
    assert total_checked == model.num_params()


def test_backward_deterministic_and_zeroable():
    model = make_model(tie_weights=True)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 11, size=(3, 8))
    targets = rng.integers(0, 11, size=(3, 8))

    model.zero_grad()
    _, l1 = model.forward(idx, targets)
    model.backward()
    g1 = {k: v.copy() for k, v in model.grads.items()}

    model.zero_grad()
    _, l2 = model.forward(idx, targets)
    model.backward()
    for k in g1:
        np.testing.assert_array_equal(g1[k], model.grads[k])
    assert l1 == l2


def test_dropout_seed_reproducible():
    m1 = make_model(tie_weights=True, dropout=0.5)
    m2 = make_model(tie_weights=True, dropout=0.5)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 11, size=(2, 8))
    targets = rng.integers(0, 11, size=(2, 8))
    _, la = m1.forward(idx, targets)
    _, lb = m2.forward(idx, targets)
    assert la == lb  # same seed -> same dropout masks


def test_eval_mode_disables_dropout():
    model = make_model(tie_weights=True, dropout=0.5)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 11, size=(2, 8))
    model.eval_mode()
    _, la = model.forward(idx, idx)
    _, lb = model.forward(idx, idx)
    assert la == lb


def test_loss_decreases_one_step_on_perfect_signal():
    """After one big step the model should fit a constant-next-token dataset."""
    cfg = ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1,
                      n_embd=16, ffn_mult=2, tie_weights=True)
    model = TinyLM(cfg, seed=0, dtype=np.float64)
    from tinyllm.optimizer import AdamW, clip_grad_global_norm
    opt = AdamW(model.params, model.grads, lr=0.05, weight_decay=0.0)
    idx = np.ones((4, 4), dtype=np.int64)
    targets = np.full((4, 4), 2, dtype=np.int64)
    _, l0 = model.forward(idx, targets)
    for _ in range(30):
        model.zero_grad()
        _, loss = model.forward(idx, targets)
        model.backward()
        clip_grad_global_norm(model.grads, 1.0)
        opt.step()
    _, l1 = model.forward(idx, targets)
    assert l1 < l0 * 0.1
