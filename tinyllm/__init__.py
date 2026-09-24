"""TinyLLM: a tiny GPT-style language model in pure NumPy.

No PyTorch, no autograd -- all forward and backward passes are implemented
by hand with NumPy linear algebra.
"""

from .config import ModelConfig
from .model import TinyLM, sample_token
from .optimizer import AdamW, clip_grad_global_norm
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "TinyLM",
    "AdamW",
    "clip_grad_global_norm",
    "CharTokenizer",
    "sample_token",
    "__version__",
]
