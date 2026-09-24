"""End-to-end smoke test: train a tiny model, sample, save, reload."""

import numpy as np

from tinyllm.config import ModelConfig
from tinyllm.data import cosine_lr, get_batch, train_val_split
from tinyllm.model import TinyLM
from tinyllm.optimizer import AdamW, clip_grad_global_norm
from tinyllm.tokenizer import CharTokenizer


def test_train_generate_save_load(tmp_path):
    text = ("the cat sat on the mat. the dog sat on the log. " * 60)
    tok = CharTokenizer.fit(text)
    ids = np.asarray(tok.encode(text), dtype=np.int16)
    train_ids, val_ids = train_val_split(ids, val_frac=0.1)
    rng = np.random.default_rng(0)

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2,
                      n_head=2, n_embd=32, ffn_mult=2, tie_weights=True)
    model = TinyLM(cfg, seed=0)
    opt = AdamW(model.params, model.grads, lr=3e-3, weight_decay=0.01)

    _, first_loss = model.forward(*get_batch(train_ids, 8, cfg.block_size, rng))
    for step in range(150):
        opt.lr = cosine_lr(step, 150, 3e-3, warmup_steps=10)
        x, y = get_batch(train_ids, 8, cfg.block_size, rng)
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        clip_grad_global_norm(model.grads, 1.0)
        opt.step()
    assert loss < first_loss - 1.0   # it is really learning

    # generation works and respects the seed
    prompt = tok.encode("the ")
    out = model.generate(prompt, max_new_tokens=20, temperature=0.7, top_k=10,
                         rng=np.random.default_rng(5))
    text_out = tok.decode(out)
    assert text_out.startswith("the ")
    out2 = model.generate(prompt, max_new_tokens=20, temperature=0.7, top_k=10,
                          rng=np.random.default_rng(5))
    assert tok.decode(out2) == text_out

    # save / load round trip reproduces identical logits
    model.save(str(tmp_path / "ckpt"))
    model2 = TinyLM.load(str(tmp_path / "ckpt"))
    xb = np.asarray([prompt[:4]], dtype=np.int64)
    la, _ = model.forward(xb)
    lb, _ = model2.forward(xb)
    np.testing.assert_allclose(la, lb, rtol=1e-5, atol=1e-5)


def test_context_cropping_respects_block_size():
    cfg = ModelConfig(vocab_size=10, block_size=8, n_layer=1, n_head=1, n_embd=16)
    model = TinyLM(cfg, seed=0)
    ids = list(np.random.default_rng(0).integers(0, 10, size=100))
    out = model.generate(ids, max_new_tokens=3, rng=np.random.default_rng(0))
    assert len(out) == 103


def test_greedy_is_deterministic():
    from tinyllm.model import sample_token
    logits = np.array([0.1, 2.0, -1.0, 0.3])
    assert sample_token(logits, temperature=0.0) == 1
    rng = np.random.default_rng(0)
    draws = {sample_token(logits, temperature=1.0, rng=rng) for _ in range(50)}
    assert draws == {0, 1, 2, 3}  # all tokens reachable in sampling mode
