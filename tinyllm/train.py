"""Train TinyLLM on a text file with pure NumPy.

Example::

    python -m tinyllm.train --data data/sample.txt --out-dir checkpoints/demo \
        --n-layer 4 --n-head 4 --n-embd 128 --block-size 128 --max-steps 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import ModelConfig
from .data import cosine_lr, get_batch, load_text, to_ids, train_val_split
from .model import TinyLM, sample_token
from .optimizer import AdamW, clip_grad_global_norm
from .tokenizer import CharTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a tiny GPT in pure NumPy")
    p.add_argument("--data", type=str, default="data/sample.txt", help="path to input text")
    p.add_argument("--out-dir", type=str, default="checkpoints/run", help="checkpoint dir")
    # model
    p.add_argument("--block-size", type=int, default=128, help="context length")
    p.add_argument("--n-layer", type=int, default=4, help="transformer blocks")
    p.add_argument("--n-head", type=int, default=4, help="attention heads")
    p.add_argument("--n-embd", type=int, default=128, help="embedding width")
    p.add_argument("--ffn-mult", type=int, default=4, help="MLP hidden multiplier")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout probability")
    p.add_argument("--no-tie-weights", action="store_true", help="separate LM head")
    # optimization
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    # bookkeeping
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-at-end", type=int, default=200, help="chars to sample at end (0=off)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    return p


def evaluate(model: TinyLM, data: np.ndarray, batch_size: int, block_size: int,
             iters: int, rng: np.random.Generator) -> float:
    model.eval_mode()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, rng)
        _, loss = model.forward(x, y)
        losses.append(loss)
    model.train_mode()
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    dtype = np.float64 if args.dtype == "float64" else np.float32
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------- data
    text = load_text(args.data)
    tokenizer = CharTokenizer.fit(text)
    ids = to_ids(text, tokenizer)
    train_ids, val_ids = train_val_split(ids, args.val_frac)
    print(f"data: {len(text):,} chars | vocab {tokenizer.vocab_size} | "
          f"train {len(train_ids):,} / val {len(val_ids):,} tokens")

    # ------------------------------------------------------------ model
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
    )
    model = TinyLM(cfg, seed=args.seed, dtype=dtype)
    opt = AdamW(model.params, model.grads, lr=args.lr, betas=(0.9, args.beta2),
                weight_decay=args.weight_decay)
    print(f"model: {cfg.n_layer} layers | {cfg.n_head} heads | {cfg.n_embd} dims | "
          f"{cfg.block_size} context | {model.num_params():,} params "
          f"({model.num_params(non_embedding=True):,} non-embedding)")
    print(f"training: {args.max_steps} steps | batch {args.batch_size} | "
          f"lr {args.lr:g} | wd {args.weight_decay:g} | dtype {args.dtype}")

    # ------------------------------------------------------------- loop
    best_val = float("inf")
    ema_loss = None
    t0 = time.time()
    tokens_seen = 0
    for step in range(args.max_steps):
        opt.lr = cosine_lr(step, args.max_steps, args.lr, args.warmup_steps,
                           args.min_lr_frac)
        x, y = get_batch(train_ids, args.batch_size, cfg.block_size, rng)
        model.train_mode()
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        gnorm = clip_grad_global_norm(model.grads, args.grad_clip)
        opt.step()
        tokens_seen += x.size

        ema_loss = loss if ema_loss is None else 0.9 * ema_loss + 0.1 * loss

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = evaluate(model, val_ids, args.batch_size, cfg.block_size,
                                args.eval_iters, rng)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.max_steps} | train {loss:.4f} "
                  f"(ema {ema_loss:.4f}) | val {val_loss:.4f} | lr {opt.lr:.2e} "
                  f"| |g| {gnorm:.2f} | {tokens_seen / max(elapsed, 1e-9):,.0f} tok/s",
                  flush=True)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            meta = dict(step=step, train_loss=loss, val_loss=val_loss,
                        best_val_loss=best_val, is_best=is_best, args=vars(args))
            model.save(args.out_dir + "/last", meta=meta)      # always keep the latest
            tokenizer.save(args.out_dir + "/last/tokenizer.json")
            if is_best:
                model.save(args.out_dir + "/best", meta=meta)  # and the best so far
                tokenizer.save(args.out_dir + "/best/tokenizer.json")
                print(f"  -> new best val loss {best_val:.4f} (saved to {args.out_dir}/best)")

        elif step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.max_steps} | train {loss:.4f} "
                  f"(ema {ema_loss:.4f}) | lr {opt.lr:.2e} | |g| {gnorm:.2f} "
                  f"| {tokens_seen / max(elapsed, 1e-9):,.0f} tok/s", flush=True)

    # -------------------------------------------------------------- done
    print(f"done in {time.time() - t0:.1f}s | best val loss {best_val:.4f}")
    if args.sample_at_end > 0:
        newline_id = tokenizer.stoi.get("\n", 0)
        start = [newline_id]
        out_ids = model.generate(start, args.sample_at_end,
                                 temperature=0.8, top_k=max(1, tokenizer.vocab_size // 2),
                                 rng=np.random.default_rng(args.seed))
        print("--- sample ---")
        print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
