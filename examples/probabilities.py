"""Peek inside the model: next-character probabilities for a prompt.

An LLM is just a probability distribution over the next token. This script
shows the top predictions for each next position -- useful for building
intuition about what the model has learned.

Usage::

    python examples/probabilities.py --ckpt checkpoints/demo/best --prompt "The cat"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyllm import TinyLM
from tinyllm.tokenizer import CharTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Show next-character probabilities")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="The ")
    p.add_argument("--top", type=int, default=5, help="how many candidates to show")
    args = p.parse_args()

    model = TinyLM.load(args.ckpt)
    tokenizer = CharTokenizer.load(args.ckpt + "/tokenizer.json")
    ids = tokenizer.encode(args.prompt)

    print(f"prompt: {args.prompt!r}\n")
    ctx = np.asarray(ids, dtype=np.int64)[None, :]
    logits, _ = model.forward(ctx)
    for pos in range(len(ids)):
        probs = np.exp(logits[0, pos].astype(np.float64))
        probs /= probs.sum()
        order = np.argsort(probs)[::-1][: args.top]
        pretty = "  ".join(f"{tokenizer.itos[i]!r}:{probs[i]:.2f}" for i in order)
        print(f"after {args.prompt[:pos + 1]!r:<22} -> {pretty}")


if __name__ == "__main__":
    main()
