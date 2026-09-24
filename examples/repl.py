"""Interactive completion REPL: type a prompt, the model continues it.

Usage (from the repo root)::

    python examples/repl.py --ckpt checkpoints/demo/best
    python examples/repl.py --ckpt checkpoints/demo/best --temperature 0.6 --top-k 10
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
    p = argparse.ArgumentParser(description="Interactive text completion with TinyLLM")
    p.add_argument("--ckpt", required=True, help="checkpoint directory (from training)")
    p.add_argument("--max-tokens", type=int, default=200, help="characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=None, help="fix for reproducible output")
    args = p.parse_args()

    model = TinyLM.load(args.ckpt)
    tokenizer = CharTokenizer.load(args.ckpt + "/tokenizer.json")
    rng = np.random.default_rng(args.seed)
    print(f"TinyLLM REPL | {model.num_params():,} params | vocab {tokenizer.vocab_size} "
          f"| context {model.cfg.block_size}")
    print("type a prompt and press Enter (Ctrl-D or Ctrl-C to quit)\n")

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        try:
            ids = tokenizer.encode(prompt)
        except KeyError as e:
            print(f"  ({e} is not in the model's vocabulary - it only knows the "
                  f"characters of its training text)")
            continue
        out = model.generate(ids, args.max_tokens, temperature=args.temperature,
                             top_k=args.top_k, rng=rng)
        print("┈" * 46)
        print(tokenizer.decode(out[len(ids):]))   # print only the continuation
        print("┈" * 46)


if __name__ == "__main__":
    main()
