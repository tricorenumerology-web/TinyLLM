"""Tests for the character tokenizer."""

import pytest

from tinyllm.tokenizer import CharTokenizer


def test_roundtrip():
    text = "hello, wörld! 123\n"
    tok = CharTokenizer.fit(text)
    ids = tok.encode(text)
    assert tok.vocab_size == len(set(text))
    assert tok.decode(ids) == text


def test_unknown_char_raises():
    tok = CharTokenizer.fit("abc")
    with pytest.raises(KeyError):
        tok.encode("d")


def test_save_load(tmp_path):
    tok = CharTokenizer.fit("hello world")
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = CharTokenizer.load(path)
    assert tok2.chars == tok.chars
    assert tok2.encode("hello") == tok.encode("hello")
    assert tok2.decode(tok2.encode("world")) == "world"
