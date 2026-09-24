# TinyLLM

A tiny GPT-style language model written **from scratch in pure NumPy** — no PyTorch,
no TensorFlow, no autograd library. Every forward pass *and* every backward pass
(back-propagation) is implemented by hand with NumPy linear algebra, in ~700 lines
of readable Python.

```text
python -m tinyllm.train --data data/sample.txt --max-steps 900 ...
step     0/900 | train 3.9553 | val 3.9461
step   400/900 | train 1.5051 | val 2.0265   <- best
step   899/900 | train 0.7234 | val 2.5471
done in 142.5s | best val loss 2.0265
```

```text
python -m tinyllm.generate --ckpt checkpoints/demo/best --prompt "Once upon a time" ...
Once upon a time the lighthouse keeper lit the lamp and the light swept the
water once, twice, three times ...
```

## Features

- **Decoder-only transformer** (GPT-2 style): token + learned positional
  embeddings, pre-norm blocks with **multi-head causal self-attention** and
  **GELU MLPs**, residual connections, final LayerNorm, optional **weight tying**
- **Hand-derived backprop** for every layer — validated to float64 precision by a
  full numerical gradient check (see `tests/test_model_gradients.py`)
- **AdamW** optimizer with decoupled weight decay, global gradient-norm clipping,
  linear warmup + cosine LR schedule
- **Character tokenizer** (no external tokenizer dependency)
- Temperature / top-k / top-p (nucleus) / greedy sampling
- Checkpointing (best-by-val + last), resumed inference via `tinyllm.generate`
- Optional dropout; float32 by default, float64 for gradient checking

## Quickstart

```bash
pip install numpy            # the only dependency

# 1) train on the bundled demo corpus (~2.5 min on a 2-core machine)
python -m tinyllm.train --data data/sample.txt --out-dir checkpoints/demo \
    --block-size 64 --n-layer 2 --n-head 2 --n-embd 64 --batch-size 16 \
    --max-steps 900 --lr 3e-3

# 2) generate text from the best checkpoint
python -m tinyllm.generate --ckpt checkpoints/demo/best \
    --prompt "Once upon a time" --max-tokens 400 --temperature 0.8 --top-k 20
```

The bundled `data/sample.txt` (~6.5 KB) is a tiny smoke-test corpus. For a real
run, download the classic ~1 MB tinyshakespeare corpus:

```bash
python scripts/download_data.py                       # needs internet
python -m tinyllm.train --data data/tinyshakespeare/input.txt \
    --out-dir checkpoints/shakespeare --max-steps 2000
```

On a 2-core CPU the demo config trains at ~7,000 tokens/s; a modern laptop
(8+ cores) is several times faster. NumPy dispatches to BLAS, so larger
models scale reasonably — this is a learning tool, not a speed demon.

## Architecture

```
            tokens (B, T)
                 │
   ┌─────────────▼──────────────┐
   │  token embedding  (V → C)  │
   │  + position embedding      │
   ├────────────────────────────┤
   │  × N blocks:               │
   │    x = x + Attn(LN1(x))    │   multi-head causal self-attention
   │    x = x + MLP(LN2(x))     │   C → 4C → C, GELU
   ├────────────────────────────┤
   │  LayerNorm                 │
   │  LM head (C → V)           │   weights tied to the embedding by default
   └─────────────┬──────────────┘
                 ▼
        softmax cross-entropy over next tokens
```

Default configuration: `n_layer=4, n_head=4, n_embd=128, block_size=128`
(~800 K parameters with a 52-char vocabulary).

## How the backprop works

There is no autograd. Each layer implements `y = f(x)` and its exact
vector-Jacobian product `dx = (∂f/∂x)ᵀ · dy`, plus parameter gradients:

| Layer | Backward math |
|---|---|
| Linear `y = xW` | `dx = dy·Wᵀ`, `dW = xᵀ·dy` |
| LayerNorm | `dx = r·(d·γ − mean(d·γ) − x̂·mean(d·γ·x̂))` with `r = 1/√(σ²+ε)` |
| Attention scores `s = (QKᵀ)/√h` | `dQ = dS·K/√h`, `dK = dSᵀ·Q/√h` |
| Softmax | `dS = P ⊙ (dP − Σ(dP ⊙ P))` |
| GELU (tanh approx.) | analytic derivative of `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| Cross-entropy | `dlogits = (softmax − onehot) / N` |
| Embedding | scatter-add of input gradients (`np.add.at`) |

The causal mask sets future scores to `-inf` before the softmax; masked
positions get probability 0, so their gradient is exactly 0 and the masking
needs no special case in backward. Residual streams simply add the upstream
gradient to the branch gradient. The whole backward pass is those per-layer
backward functions composed in reverse order (`TinyLM.backward`).

Correctness is proven, not assumed: `tests/test_model_gradients.py` perturbs
**every single parameter** with central finite differences and matches the
analytic gradients to a relative error < 1e-6 in float64.

## Project layout

```
tinyllm/
  config.py      ModelConfig dataclass + (de)serialization
  tokenizer.py   character-level tokenizer
  model.py       transformer forward/backward + sampling
  optimizer.py   AdamW + gradient clipping
  data.py        batching, train/val split, LR schedule
  train.py       CLI training loop
  generate.py    CLI text generation
data/sample.txt  bundled demo corpus
scripts/         dataset downloader
tests/           gradient checks + unit + end-to-end tests
```

## Command-line reference

Training (`python -m tinyllm.train -h` for all flags):

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/sample.txt` | training text file |
| `--out-dir` | `checkpoints/run` | writes `<out>/best` and `<out>/last` |
| `--n-layer/--n-head/--n-embd` | 4 / 4 / 128 | transformer size |
| `--block-size` | 128 | context length in tokens |
| `--max-steps/--batch-size` | 2000 / 32 | optimization budget |
| `--lr/--warmup-steps/--weight-decay` | 3e-3 / 50 / 0.01 | AdamW settings |
| `--dropout` | 0.0 | regularization (useful on small corpora) |
| `--no-tie-weights` | off | use a separate LM head matrix |

Generation: `--ckpt`, `--prompt`, `--max-tokens`, `--temperature`,
`--top-k`, `--top-p`, `--greedy`, `--seed`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

Covers: full-model numerical gradient check (tied and untied), tokenizer
round-trip and persistence, AdamW/decay/clipping, deterministic sampling,
dropout modes, and a train → sample → save → reload end-to-end test.

## Things you can try next

- Train on your own text: `python -m tinyllm.train --data your.txt ...`
  (any plain-text file works; the tokenizer adapts to its alphabet)
- Scale up: `--n-embd 256 --n-layer 6 --n-head 8` (keep an eye on step time)
- Add an interactive REPL around `generate`, a KV-cache for faster sampling,
  or a BPE tokenizer — the interfaces (`TinyLM`, `CharTokenizer`) are small
  and self-contained on purpose.
