from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from .encoder import encode, encode_with_phrases
from .online_manifest import ROUTE_LITERAL, ROUTE_PHRASE
from .predictor.phrases import PhraseTable
from .sideinfo_codec import pack_string, unpack_string

RESIDUAL_RAW = 0
RESIDUAL_LITERAL = 1
RESIDUAL_PHRASE = 2
RESIDUAL_TYPED = 3
_TEMPLATE_SPAN_FMT = "<III"
_TEMPLATE_SPAN_SIZE = struct.calcsize(_TEMPLATE_SPAN_FMT)
_RESIDUAL_MAGIC = b"RSD1"

# Type prefixes for typed residual payloads (single byte discriminators)
_TYPED_MODE_ENUM = 0x01
_TYPED_MODE_VERSION = 0x02
_TYPED_MODE_PATH_OR_URL = 0x03
_TYPED_MODE_NUMBER_WITH_UNIT = 0x04
_TYPED_MODE_IDENTIFIER = 0x05

_NUMBER_WITH_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([A-Za-z%]+$)")


@dataclass(frozen=True)
class ResidualSegment:
    original_start: int
    original_end: int
    text: str
    route: str
    payload: bytes
    residual_type: str = ""  # "" for literal/phrase, "version"/"enum"/etc for typed

    @property
    def encoded_bytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class EncodedResidual:
    segments: tuple[ResidualSegment, ...]

    @property
    def total_payload_bytes(self) -> int:
        return sum(segment.encoded_bytes for segment in self.segments)

    def serialize(self) -> bytes:
        parts = [_RESIDUAL_MAGIC, struct.pack("<H", len(self.segments))]
        for segment in self.segments:
            route_id = _route_to_id(segment.route)
            parts.append(
                struct.pack(
                    _TEMPLATE_SPAN_FMT,
                    segment.original_start,
                    segment.original_end,
                    route_id,
                )
            )
            parts.append(struct.pack("<I", len(segment.payload)))
            parts.append(segment.payload)
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncodedResidual":
        if not data:
            return cls(segments=())
        if not data.startswith(_RESIDUAL_MAGIC):
            raise ValueError("invalid residual payload")
        offset = len(_RESIDUAL_MAGIC)
        if offset + 2 > len(data):
            raise ValueError("invalid residual payload")
        count = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        segments: list[ResidualSegment] = []
        for _ in range(count):
            if offset + _TEMPLATE_SPAN_SIZE + 4 > len(data):
                raise ValueError("invalid residual payload")
            start, end, route_id = struct.unpack(
                _TEMPLATE_SPAN_FMT,
                data[offset:offset + _TEMPLATE_SPAN_SIZE],
            )
            offset += _TEMPLATE_SPAN_SIZE
            payload_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            payload_end = offset + payload_len
            if payload_end > len(data):
                raise ValueError("invalid residual payload")
            payload = data[offset:payload_end]
            offset = payload_end
            segments.append(
                ResidualSegment(
                    original_start=start,
                    original_end=end,
                    text="",
                    route=_id_to_route(route_id),
                    payload=payload,
                )
            )
        if offset != len(data):
            raise ValueError("invalid residual payload")
        return cls(segments=tuple(segments))


def encode_residual_segments(
    text: str,
    spans: tuple[tuple[int, int], ...],
    phrase_set: frozenset[str],
    priors: dict[str, float] | None,
    max_order: int,
    slot_hints: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> EncodedResidual:
    segments: list[ResidualSegment] = []
    for i, (start, end) in enumerate(spans):
        snippet = text[start:end]
        field, enum_candidates = slot_hints[i] if i < len(slot_hints) else ("", ())

        # Try typed residual encoding first
        typed_result = _try_encode_typed_residual(snippet, field, enum_candidates)
        if typed_result is not None:
            residual_type, payload = typed_result
            segments.append(
                ResidualSegment(
                    original_start=start,
                    original_end=end,
                    text=snippet,
                    route="typed",
                    payload=payload,
                    residual_type=residual_type,
                )
            )
            continue

        # Fallback to literal/phrase
        literal_payload = encode(snippet, priors=priors, max_order=max_order)
        route = ROUTE_LITERAL
        payload = literal_payload
        if phrase_set and len(snippet) >= 8:
            phrase_payload = encode_with_phrases(
                snippet,
                phrase_set,
                priors=priors,
                max_order=max_order,
            )
            if len(phrase_payload) < len(literal_payload):
                route = ROUTE_PHRASE
                payload = phrase_payload
        segments.append(
            ResidualSegment(
                original_start=start,
                original_end=end,
                text=snippet,
                route=route,
                payload=payload,
            )
        )
    return EncodedResidual(segments=tuple(segments))


def serialize_phrase_table_for_template(phrases: tuple[str, ...]) -> bytes:
    return PhraseTable(phrases=phrases).serialize()


def serialize_string_tuple(items: tuple[str, ...]) -> bytes:
    parts = [struct.pack("<H", len(items))]
    for item in items:
        parts.append(pack_string(item))
    return b"".join(parts)


def deserialize_string_tuple(data: bytes) -> tuple[str, ...]:
    if not data:
        return ()
    if len(data) < 2:
        raise ValueError("invalid tuple payload")
    count = struct.unpack("<H", data[:2])[0]
    offset = 2
    items: list[str] = []
    for _ in range(count):
        item, offset = unpack_string(data, offset)
        items.append(item)
    if offset != len(data):
        raise ValueError("invalid tuple payload")
    return tuple(items)


def _route_to_id(route: str) -> int:
    if route == ROUTE_PHRASE:
        return RESIDUAL_PHRASE
    if route == ROUTE_LITERAL:
        return RESIDUAL_LITERAL
    if route == "typed":
        return RESIDUAL_TYPED
    return RESIDUAL_RAW


def _id_to_route(route_id: int) -> str:
    if route_id == RESIDUAL_PHRASE:
        return ROUTE_PHRASE
    if route_id == RESIDUAL_TYPED:
        return "typed"
    return ROUTE_LITERAL


# ------------------------------------------------------------------
# Typed residual encoding/decoding helpers
# ------------------------------------------------------------------


def _try_encode_typed_residual(
    text: str,
    field: str,
    enum_candidates: tuple[str, ...],
) -> tuple[str, bytes] | None:
    """Try to encode residual text as a typed value. Returns (type, payload) or None."""
    stripped = text.strip()
    if not stripped:
        return None

    # Enum: check if value is in candidate list
    if enum_candidates and stripped in enum_candidates:
        idx = enum_candidates.index(stripped)
        if idx < 256:
            return "enum", bytes([_TYPED_MODE_ENUM, idx])

    # Version: looks like X.Y.Z
    if _looks_like_version(stripped):
        return "version", bytes([_TYPED_MODE_VERSION]) + _encode_version_simple(stripped)

    # Path or URL: starts with /, http, https, ftp, file, C:/, etc.
    if _looks_like_path_or_url(stripped):
        return "path_or_url", bytes([_TYPED_MODE_PATH_OR_URL]) + _encode_path_simple(stripped)

    # Number with unit: e.g. "100ms", "3.14kg", "50%"
    if _NUMBER_WITH_UNIT_RE.match(stripped):
        return "number_with_unit", bytes([_TYPED_MODE_NUMBER_WITH_UNIT]) + _encode_number_simple(stripped)

    # Identifier: alphanumeric with underscores/hyphens, no spaces
    if _looks_like_identifier(stripped):
        encoded = stripped.encode("utf-8")
        return "identifier", bytes([_TYPED_MODE_IDENTIFIER, len(encoded)]) + encoded

    return None


def _decode_typed_residual(payload: bytes, residual_type: str) -> str:
    """Decode a typed residual payload back to text."""
    if not payload:
        return ""

    mode = payload[0]
    body = payload[1:]

    if mode == _TYPED_MODE_ENUM:
        # mode byte + index byte (index is resolved by caller with enum_candidates)
        # For standalone decode, we return a placeholder since we don't have candidates
        return f"<enum:{body[0]}>"

    if mode == _TYPED_MODE_VERSION:
        return _decode_version_simple(body)

    if mode == _TYPED_MODE_PATH_OR_URL:
        return _decode_path_simple(body)

    if mode == _TYPED_MODE_NUMBER_WITH_UNIT:
        return _decode_number_simple(body)

    if mode == _TYPED_MODE_IDENTIFIER:
        length = body[0]
        return body[1:1 + length].decode("utf-8")

    # Fallback: treat as raw UTF-8
    return payload.decode("utf-8", errors="replace")


def _looks_like_version(text: str) -> bool:
    """Check if text looks like a version string (e.g. v1.2.3, 1.0.0)."""
    return bool(re.match(r"^(?:v)?\d+(?:\.\d+){1,3}(?:[-.][a-zA-Z0-9]+)?$", text))


def _encode_version_simple(value: str) -> bytes:
    """Simple version encoding: raw UTF-8 (full encoding in template_codec)."""
    return value.encode("utf-8")


def _decode_version_simple(payload: bytes) -> str:
    return payload.decode("utf-8")


def _looks_like_path_or_url(text: str) -> bool:
    """Check if text looks like a path or URL."""
    return text.startswith(("/", "http://", "https://", "ftp://", "file://")) or bool(re.match(r"^[A-Z]:[/\\]", text))


def _encode_path_simple(value: str) -> bytes:
    """Simple path encoding: raw UTF-8 (full encoding with prefix index in template_codec)."""
    return value.encode("utf-8")


def _decode_path_simple(payload: bytes) -> str:
    return payload.decode("utf-8")


def _looks_like_number_with_unit(text: str) -> bool:
    """Check if text looks like a number with unit."""
    return bool(_NUMBER_WITH_UNIT_RE.match(text))


def _encode_number_simple(value: str) -> bytes:
    """Simple number-with-unit encoding: raw UTF-8."""
    return value.encode("utf-8")


def _decode_number_simple(payload: bytes) -> str:
    return payload.decode("utf-8")


def _looks_like_identifier(text: str) -> bool:
    """Check if text looks like an identifier (alphanumeric + underscore/hyphen, no spaces)."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$", text)) and " " not in text
