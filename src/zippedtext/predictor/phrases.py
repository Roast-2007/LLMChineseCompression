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

from ..sideinfo_codec import pack_string, unpack_string

_PHRASE_TABLE_MAGIC = b"PHT1"


@dataclass(frozen=True)
class PhraseTable:
    """Ordered collection of phrases for encoding."""

    phrases: tuple[str, ...]

    def serialize(self) -> bytes:
        parts = [_PHRASE_TABLE_MAGIC, struct.pack("<H", len(self.phrases))]
        for phrase in self.phrases:
            parts.append(pack_string(phrase))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "PhraseTable":
        """Reconstruct from serialized bytes."""
        if not data:
            return cls(phrases=())
        if data.startswith(_PHRASE_TABLE_MAGIC):
            return _deserialize_length_prefixed(data)
        return _deserialize_legacy_null_terminated(data)


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
    freq: Counter[str] = Counter()
    for length in range(min_len, max_len + 1):
        for i in range(len(text) - length + 1):
            sub = text[i:i + length]
            freq[sub] += 1

    candidates = {s: c for s, c in freq.items() if c >= min_freq}

    scored = [
        (s, c, (c - 1) * (len(s) - 1))
        for s, c in candidates.items()
    ]
    scored.sort(key=lambda t: t[2], reverse=True)

    selected: list[str] = []
    selected_set: set[str] = set()
    for phrase, _count, _score in scored:
        if len(selected) >= max_phrases:
            break
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
    end_limit = min(pos + max_len, len(text))
    for length in range(end_limit - pos, 1, -1):
        candidate = text[pos:pos + length]
        if candidate in phrase_set:
            return candidate
    return None


def _deserialize_length_prefixed(data: bytes) -> PhraseTable:
    if len(data) < len(_PHRASE_TABLE_MAGIC) + 2:
        raise ValueError("invalid phrase table: truncated header")
    count = struct.unpack("<H", data[len(_PHRASE_TABLE_MAGIC):len(_PHRASE_TABLE_MAGIC) + 2])[0]
    offset = len(_PHRASE_TABLE_MAGIC) + 2
    phrases: list[str] = []
    for _ in range(count):
        phrase, offset = unpack_string(data, offset)
        phrases.append(phrase)
    if offset != len(data):
        raise ValueError("invalid phrase table: trailing bytes")
    return PhraseTable(phrases=tuple(phrases))


def _deserialize_legacy_null_terminated(data: bytes) -> PhraseTable:
    if len(data) < 2:
        return PhraseTable(phrases=())
    count = struct.unpack("<H", data[:2])[0]
    offset = 2
    phrases: list[str] = []
    for _ in range(count):
        end = data.index(b"\x00", offset)
        phrases.append(data[offset:end].decode("utf-8"))
        offset = end + 1
    return PhraseTable(phrases=tuple(phrases))
