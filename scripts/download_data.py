"""Download the classic tinyshakespeare corpus (~1.1 MB) for training.

Requires internet access::

    python scripts/download_data.py
"""

from __future__ import annotations

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare")
OUT_FILE = os.path.join(OUT_DIR, "input.txt")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, OUT_FILE)
    size = os.path.getsize(OUT_FILE)
    print(f"saved {size:,} bytes to {os.path.relpath(OUT_FILE)}")
    print("train with: python -m tinyllm.train --data data/tinyshakespeare/input.txt")


if __name__ == "__main__":
    main()
