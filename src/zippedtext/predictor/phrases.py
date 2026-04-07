"""Phrase-level encoding: detect and encode high-frequency multi-character phrases.

Two-pass scheme:
  1. Analysis: scan text for frequent substrings (2-8 chars), select top-N.
  2. Encoding: greedy longest-match at each position; matched phrases are
     encoded as single symbols in the adaptive predictor.

The phrase table is serialized into the .ztxt file so the decoder can
reconstruct identical vocabulary.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PhraseTable:
    """Ordered collection of phrases for encoding."""
    phrases: tuple[str, ...]

    def serialize(self) -> bytes:
        """Pack into bytes: [uint16 count] [utf8 phrase + \\x00] ..."""
        parts = [struct.pack("<H", len(self.phrases))]
        for p in self.phrases:
            parts.append(p.encode("utf-8") + b"\x00")
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "PhraseTable":
        """Reconstruct from serialized bytes."""
        if len(data) < 2:
            return cls(phrases=())
        count = struct.unpack("<H", data[:2])[0]
        offset = 2
        phrases: list[str] = []
        for _ in range(count):
            end = data.index(b"\x00", offset)
            phrases.append(data[offset:end].decode("utf-8"))
            offset = end + 1
        return cls(phrases=tuple(phrases))


def build_phrase_table(
    text: str,
    max_phrases: int = 128,
    min_freq: int = 3,
    min_len: int = 2,
    max_len: int = 8,
) -> PhraseTable:
    """Scan *text* and select the most valuable phrases.

    "Value" is estimated as ``(freq - 1) * (len - 1)`` — the number of
    extra characters saved by treating the phrase as a single symbol.
    """
    # Count all substrings of length 2..max_len
    freq: Counter[str] = Counter()
    for length in range(min_len, max_len + 1):
        for i in range(len(text) - length + 1):
            sub = text[i:i + length]
            freq[sub] += 1

    # Filter by minimum frequency
    candidates = {s: c for s, c in freq.items() if c >= min_freq}

    # Score: estimated bits saved
    scored = [
        (s, c, (c - 1) * (len(s) - 1))
        for s, c in candidates.items()
    ]
    scored.sort(key=lambda t: t[2], reverse=True)

    # Remove phrases that are substrings of higher-scoring phrases
    # (greedy: a longer phrase already covers its substrings)
    selected: list[str] = []
    selected_set: set[str] = set()
    for phrase, _count, _score in scored:
        if len(selected) >= max_phrases:
            break
        # Skip if this phrase is a substring of an already-selected phrase
        is_covered = any(phrase in s and phrase != s for s in selected_set)
        if not is_covered:
            selected.append(phrase)
            selected_set.add(phrase)

    return PhraseTable(phrases=tuple(selected))


def greedy_phrase_match(
    text: str,
    pos: int,
    phrase_set: frozenset[str],
    max_len: int = 8,
) -> str | None:
    """At position *pos*, find the longest phrase that matches.

    Returns the matched phrase string, or ``None`` if no phrase matches.
    """
    # Try longest first for greedy matching
    end_limit = min(pos + max_len, len(text))
    for length in range(end_limit - pos, 1, -1):
        candidate = text[pos:pos + length]
        if candidate in phrase_set:
            return candidate
    return None
