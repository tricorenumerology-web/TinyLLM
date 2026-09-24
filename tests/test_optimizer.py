"""Tests for AdamW and gradient clipping."""

import numpy as np

from tinyllm.optimizer import AdamW, clip_grad_global_norm


def test_adamw_minimizes_quadratic():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((6, 6))
    a = m @ m.T / 6.0 + np.eye(6)                        # symmetric positive definite
    x = np.ones(6)
    grads = {"x": np.zeros_like(x)}
    opt = AdamW({"x": x}, grads, lr=0.05, betas=(0.9, 0.99), weight_decay=0.0)
    f0 = x @ a @ x
    for _ in range(1500):
        grads["x"][...] = 2 * a @ x                      # grad of x^T A x
        opt.step()
    # optimum is the origin; Adam with a fixed lr oscillates in a ~lr ball
    assert np.abs(x).max() < 0.25
    assert x @ a @ x < 1e-3 * f0


def test_weight_decay_decoupled_and_zeroed_grad():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    w_orig = w.copy()                     # keep unmutated reference values
    params = {"w": w, "b": np.zeros(3)}
    grads = {"w": np.zeros_like(w), "b": np.zeros(3)}
    opt = AdamW(params, grads, lr=0.1, weight_decay=0.5)
    opt.step()  # grad == 0 -> only weight decay acts on the matrix
    np.testing.assert_allclose(params["w"], w_orig * (1 - 0.1 * 0.5))
    np.testing.assert_array_equal(params["b"], np.zeros(3))  # no decay on 1-D


def test_clip_grad_global_norm():
    grads = {"a": np.array([3.0, 0.0]), "b": np.array([0.0, 4.0])}
    pre = clip_grad_global_norm(grads, max_norm=1.0)
    assert pre == 5.0
    post = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    assert abs(post - 1.0) < 1e-6
    # no clipping when under the max
    grads = {"a": np.array([0.3, 0.0]), "b": np.array([0.0, 0.4])}
    pre = clip_grad_global_norm(grads, max_norm=1.0)
    assert pre == 0.5
    assert np.sqrt(sum(np.sum(g ** 2) for g in grads.values())) == 0.5
