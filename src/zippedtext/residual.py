from __future__ import annotations

import struct
from dataclasses import dataclass

from .encoder import encode, encode_with_phrases
from .online_manifest import ROUTE_LITERAL, ROUTE_PHRASE
from .predictor.phrases import PhraseTable
from .sideinfo_codec import pack_string, unpack_string

RESIDUAL_RAW = 0
RESIDUAL_LITERAL = 1
RESIDUAL_PHRASE = 2
_TEMPLATE_SPAN_FMT = "<III"
_TEMPLATE_SPAN_SIZE = struct.calcsize(_TEMPLATE_SPAN_FMT)
_RESIDUAL_MAGIC = b"RSD1"


@dataclass(frozen=True)
class ResidualSegment:
    original_start: int
    original_end: int
    text: str
    route: str
    payload: bytes

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
) -> EncodedResidual:
    segments: list[ResidualSegment] = []
    for start, end in spans:
        snippet = text[start:end]
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
    return RESIDUAL_RAW


def _id_to_route(route_id: int) -> str:
    if route_id == RESIDUAL_PHRASE:
        return ROUTE_PHRASE
    return ROUTE_LITERAL
