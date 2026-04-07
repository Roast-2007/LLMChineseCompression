"""Tests for arithmetic encoder/decoder round-trip correctness."""

import io
import random
import sys

sys.path.insert(0, "src")

from zippedtext.arithmetic import ArithmeticDecoder, ArithmeticEncoder, probs_to_cdf
from zippedtext.bitstream import BitInputStream, BitOutputStream


def roundtrip(symbols: list[int], cdfs: list[list[int]]) -> list[int]:
    """Encode then decode a sequence of symbols, return decoded symbols."""
    # Encode
    buf = io.BytesIO()
    out = BitOutputStream(buf)
    enc = ArithmeticEncoder(out)
    for sym, cdf in zip(symbols, cdfs):
        enc.encode(cdf, sym)
    enc.finish()
    compressed = buf.getvalue()

    # Decode
    inp = BitInputStream(compressed)
    dec = ArithmeticDecoder(inp)
    decoded = []
    for cdf in cdfs:
        decoded.append(dec.decode(cdf))
    return decoded


def test_simple_uniform():
    """Uniform distribution over 4 symbols."""
    vocab = 4
    probs = [0.25] * vocab
    cdf = probs_to_cdf(probs, vocab)
    symbols = [0, 1, 2, 3, 0, 1, 2, 3]
    cdfs = [cdf] * len(symbols)
    assert roundtrip(symbols, cdfs) == symbols


def test_skewed_distribution():
    """Highly skewed distribution — one dominant symbol."""
    vocab = 5
    probs = [0.9, 0.025, 0.025, 0.025, 0.025]
    cdf = probs_to_cdf(probs, vocab)
    symbols = [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 4]
    cdfs = [cdf] * len(symbols)
    assert roundtrip(symbols, cdfs) == symbols


def test_random_sequences():
    """Random symbols with random distributions."""
    rng = random.Random(42)
    for _ in range(20):
        vocab = rng.randint(2, 100)
        length = rng.randint(1, 200)
        # Random probability distribution
        raw = [rng.random() for _ in range(vocab)]
        total = sum(raw)
        probs = [r / total for r in raw]
        cdf = probs_to_cdf(probs, vocab)
        symbols = [rng.randint(0, vocab - 1) for _ in range(length)]
        cdfs = [cdf] * length
        assert roundtrip(symbols, cdfs) == symbols, f"Failed: vocab={vocab}, len={length}"


def test_varying_cdfs():
    """Each symbol position uses a different CDF (simulates LLM context changes)."""
    rng = random.Random(123)
    length = 50
    vocab = 10
    symbols = []
    cdfs = []
    for _ in range(length):
        raw = [rng.random() for _ in range(vocab)]
        total = sum(raw)
        probs = [r / total for r in raw]
        cdf = probs_to_cdf(probs, vocab)
        cdfs.append(cdf)
        symbols.append(rng.randint(0, vocab - 1))
    assert roundtrip(symbols, cdfs) == symbols


def test_large_vocab():
    """Simulate a real tokenizer vocab size (~20000)."""
    rng = random.Random(999)
    vocab = 20000
    length = 100
    raw = [rng.random() for _ in range(vocab)]
    total = sum(raw)
    probs = [r / total for r in raw]
    cdf = probs_to_cdf(probs, vocab)
    symbols = [rng.randint(0, vocab - 1) for _ in range(length)]
    cdfs = [cdf] * length
    assert roundtrip(symbols, cdfs) == symbols


def test_cdf_properties():
    """Verify CDF invariants."""
    probs = [0.5, 0.3, 0.15, 0.05]
    cdf = probs_to_cdf(probs, 4)
    assert cdf[0] == 0
    assert cdf[-1] == (1 << 32)
    for i in range(len(cdf) - 1):
        assert cdf[i + 1] > cdf[i], f"CDF not strictly increasing at {i}"


def test_single_symbol():
    """Edge case: single symbol in vocab."""
    cdf = probs_to_cdf([1.0], 1)
    symbols = [0, 0, 0]
    cdfs = [cdf] * 3
    assert roundtrip(symbols, cdfs) == symbols


if __name__ == "__main__":
    test_simple_uniform()
    test_skewed_distribution()
    test_random_sequences()
    test_varying_cdfs()
    test_large_vocab()
    test_cdf_properties()
    test_single_symbol()
    print("All arithmetic coding tests passed!")
