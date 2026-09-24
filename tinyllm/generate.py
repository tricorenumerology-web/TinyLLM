"""Generate text from a trained TinyLLM checkpoint.

Example::

    python -m tinyllm.generate --ckpt checkpoints/run \
        --prompt "Once upon a time" --max-tokens 400 --temperature 0.8 --top-k 40
"""

from __future__ import annotations

import argparse

import numpy as np

from .model import TinyLM
from .tokenizer import CharTokenizer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sample text from a TinyLLM checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint directory")
    p.add_argument("--prompt", type=str, default="\n", help="prompt text")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--greedy", action="store_true", help="argmax decoding")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    model = TinyLM.load(args.ckpt)
    tokenizer = CharTokenizer.load(args.ckpt + "/tokenizer.json")
    print(f"loaded {model.num_params():,} params | vocab {tokenizer.vocab_size} "
          f"| context {model.cfg.block_size}", flush=True)

    rng = np.random.default_rng(args.seed)
    if args.greedy:
        args.temperature, args.top_k, args.top_p = 0.0, None, None

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as e:
        raise SystemExit(f"prompt contains a character not seen in training: {e}")

    out_ids = model.generate(prompt_ids, args.max_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, rng=rng)
    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
