"""Tokenizer wrapper — uses a character-level tokenizer for Chinese/English/digits.

DeepSeek's HuggingFace tokenizer requires `transformers` which is heavy.
Instead, we use a simple character-level tokenizer that:
  - Assigns each unique character a token ID
  - Builds vocab from the input text (stored in compressed file for decode)
  - Guarantees perfect round-trip: decode(encode(text)) == text

This avoids the transformers dependency and UNK token issues entirely.
For LLM API calls, we send raw text (not token IDs) to DeepSeek.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass


@dataclass
class Vocab:
    """Character-level vocabulary with deterministic ordering."""

    char_to_id: dict[str, int]
    id_to_char: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    def serialize(self) -> bytes:
        """Serialize vocab to compact binary format.

        Format: uint16 count + raw UTF-8 characters separated by null byte.
        Much more compact than JSON for CJK characters.
        """
        parts = [len(self.id_to_char).to_bytes(2, "little")]
        parts.append("\x00".join(self.id_to_char).encode("utf-8"))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> Vocab:
        """Deserialize vocab from compact binary format."""
        count = int.from_bytes(data[:2], "little")
        chars = data[2:].decode("utf-8").split("\x00")[:count]
        char_to_id = {ch: i for i, ch in enumerate(chars)}
        return cls(char_to_id=char_to_id, id_to_char=chars)


def build_vocab(text: str) -> Vocab:
    """Build a character-level vocabulary from the given text.

    Characters are ordered by frequency (most frequent first) for better
    compression with arithmetic coding — common chars get lower IDs which
    tend to be more predictable.
    """
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # Sort by frequency descending, then by char for determinism
    sorted_chars = sorted(freq.keys(), key=lambda c: (-freq[c], c))

    char_to_id = {ch: i for i, ch in enumerate(sorted_chars)}
    return Vocab(char_to_id=char_to_id, id_to_char=sorted_chars)
