"""Uniform CDF construction utilities shared by encoder and decoder."""

from __future__ import annotations

from .arithmetic import TOTAL

_UNIFORM_CDF_CACHE: dict[int, list[int]] = {}


def uniform_cdf(n: int) -> list[int]:
    """Build a uniform CDF for *n* symbols.  Cached for common sizes."""
    cached = _UNIFORM_CDF_CACHE.get(n)
    if cached is not None:
        return cached
    step = TOTAL // n
    cdf = [i * step for i in range(n)]
    cdf.append(TOTAL)
    _UNIFORM_CDF_CACHE[n] = cdf
    return cdf
