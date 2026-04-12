"""Integer arithmetic encoder/decoder (Witten-Neal-Cleary style).

Uses 32-bit precision registers. The CDF is a list of (vocab_size + 1) integers
where cdf[0] = 0 and cdf[-1] = TOTAL = 2^PRECISION_BITS.
"""

from __future__ import annotations

import io
from typing import Sequence

from .bitstream import BitInputStream, BitOutputStream

PRECISION_BITS = 32
TOTAL = 1 << PRECISION_BITS          # 4_294_967_296
HALF = TOTAL >> 1                     # 2_147_483_648
QUARTER = TOTAL >> 2                  # 1_073_741_824
THREE_QUARTER = HALF + QUARTER        # 3_221_225_472
MASK = TOTAL - 1                      # 0xFFFFFFFF


def probs_to_cdf(probs: Sequence[float], vocab_size: int) -> list[int]:
    """Convert float probabilities to an integer CDF for arithmetic coding.

    Guarantees:
    - Every symbol gets frequency >= 1
    - cdf[0] = 0, cdf[vocab_size] = TOTAL
    - Deterministic: same input always produces same output
    """
    # Scale probabilities to integers with minimum frequency of 1
    freqs = [max(1, int(p * TOTAL)) for p in probs[:vocab_size]]

    # Pad if probs is shorter than vocab_size
    while len(freqs) < vocab_size:
        freqs.append(1)

    # Adjust the largest-frequency symbol to make sum exactly TOTAL
    diff = sum(freqs) - TOTAL
    max_idx = max(range(len(freqs)), key=lambda i: freqs[i])
    freqs[max_idx] -= diff
    # Safety: if adjustment makes it < 1, redistribute
    if freqs[max_idx] < 1:
        freqs[max_idx] = 1
        # Recompute diff and spread across other large entries
        diff = sum(freqs) - TOTAL
        for i in sorted(range(len(freqs)), key=lambda i: freqs[i], reverse=True):
            if i == max_idx:
                continue
            adj = min(diff, freqs[i] - 1)
            freqs[i] -= adj
            diff -= adj
            if diff == 0:
                break
        # Final safety: if diff > 0 (extreme edge case), force-adjust
        if diff > 0:
            freqs[max_idx] = max(1, freqs[max_idx] - diff)
            diff = 0
        # If still not balanced (shouldn't happen), force set max to close gap
        remaining = TOTAL - sum(freqs)
        if remaining != 0:
            freqs[max_idx] += remaining

    # Build cumulative distribution
    cdf = [0] * (vocab_size + 1)
    for i in range(vocab_size):
        cdf[i + 1] = cdf[i] + freqs[i]

    assert cdf[-1] == TOTAL, f"CDF sum {cdf[-1]} != {TOTAL}"
    return cdf


class ArithmeticEncoder:
    """Encodes symbols using arithmetic coding with per-symbol CDFs."""

    def __init__(self, out: BitOutputStream) -> None:
        self._out = out
        self._low = 0
        self._high = MASK  # TOTAL - 1
        self._pending = 0  # underflow bit counter

    def encode(self, cdf: list[int], symbol: int) -> None:
        """Encode one symbol given its CDF."""
        rng = self._high - self._low + 1
        self._high = self._low + (rng * cdf[symbol + 1]) // TOTAL - 1
        self._low = self._low + (rng * cdf[symbol]) // TOTAL

        # Normalize: shift out matching MSBs and handle underflow
        while True:
            if self._high < HALF:
                # Both in lower half → output 0 + pending 1s
                self._output_bit(0)
            elif self._low >= HALF:
                # Both in upper half → output 1 + pending 0s
                self._output_bit(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                # Underflow: converging near middle
                self._pending += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) | 1) & MASK

    def finish(self) -> None:
        """Flush remaining bits to finalize the encoded stream."""
        # Output enough bits to disambiguate the final interval
        self._pending += 1
        if self._low < QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)
        self._out.close()

    def _output_bit(self, bit: int) -> None:
        self._out.write_bit(bit)
        # Flush pending underflow bits (opposite of the bit just written)
        for _ in range(self._pending):
            self._out.write_bit(bit ^ 1)
        self._pending = 0


class ArithmeticDecoder:
    """Decodes symbols using arithmetic coding with per-symbol CDFs."""

    def __init__(self, inp: BitInputStream) -> None:
        self._inp = inp
        self._low = 0
        self._high = MASK
        # Initialize code register from the first PRECISION_BITS bits
        self._code = 0
        for _ in range(PRECISION_BITS):
            self._code = (self._code << 1) | self._inp.read_bit()

    def decode(self, cdf: list[int]) -> int:
        """Decode one symbol given its CDF. Returns the symbol index."""
        rng = self._high - self._low + 1
        # Determine which symbol the current code falls into
        value = ((self._code - self._low + 1) * TOTAL - 1) // rng
        # Binary search for the symbol in the CDF
        symbol = _bisect_cdf(cdf, value)

        # Update interval (same as encoder)
        self._high = self._low + (rng * cdf[symbol + 1]) // TOTAL - 1
        self._low = self._low + (rng * cdf[symbol]) // TOTAL

        # Normalize (mirror of encoder)
        while True:
            if self._high < HALF:
                pass  # both in lower half
            elif self._low >= HALF:
                self._code -= HALF
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                self._code -= QUARTER
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) | 1) & MASK
            self._code = ((self._code << 1) | self._inp.read_bit()) & MASK

        return symbol


def _bisect_cdf(cdf: list[int], value: int) -> int:
    """Binary search: find the largest i such that cdf[i] <= value."""
    lo, hi = 0, len(cdf) - 2  # symbol range: [0, vocab_size-1]
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if cdf[mid] <= value:
            lo = mid
        else:
            hi = mid - 1
    return lo
