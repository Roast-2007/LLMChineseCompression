from __future__ import annotations

from dataclasses import dataclass

from .encoder import encode, encode_with_phrases
from .online_manifest import ROUTE_LITERAL, ROUTE_PHRASE
from .segment import TextSegment

MIN_GAIN_BYTES = 4
PHRASE_ENABLED_KINDS = frozenset({"prose", "list", "table", "config", "mixed"})


@dataclass(frozen=True)
class RoutedSegment:
    segment: TextSegment
    route: str
    payload: bytes
    original_bytes: int
    encoded_bytes: int


@dataclass(frozen=True)
class RouteSummary:
    routed_segments: tuple[RoutedSegment, ...]
    route_counts: tuple[tuple[str, int], ...]


def route_segments(
    text: str,
    segments: tuple[TextSegment, ...],
    phrase_set: frozenset[str],
    priors: dict[str, float] | None,
    max_order: int,
) -> RouteSummary:
    routed: list[RoutedSegment] = []
    route_counts: dict[str, int] = {ROUTE_LITERAL: 0, ROUTE_PHRASE: 0}

    for segment in segments:
        segment_text = text[segment.start:segment.end]
        literal_payload = encode(
            segment_text,
            priors=priors,
            max_order=max_order,
        )
        literal = RoutedSegment(
            segment=segment,
            route=ROUTE_LITERAL,
            payload=literal_payload,
            original_bytes=len(segment_text.encode("utf-8")),
            encoded_bytes=len(literal_payload),
        )
        decision = literal

        if phrase_set and segment.kind in PHRASE_ENABLED_KINDS:
            phrase_payload = encode_with_phrases(
                segment_text,
                phrase_set,
                priors=priors,
                max_order=max_order,
            )
            if len(phrase_payload) + MIN_GAIN_BYTES < len(literal_payload):
                decision = RoutedSegment(
                    segment=segment,
                    route=ROUTE_PHRASE,
                    payload=phrase_payload,
                    original_bytes=len(segment_text.encode("utf-8")),
                    encoded_bytes=len(phrase_payload),
                )

        route_counts[decision.route] = route_counts.get(decision.route, 0) + 1
        routed.append(decision)

    summary = tuple(sorted(route_counts.items(), key=lambda item: item[0]))
    return RouteSummary(routed_segments=tuple(routed), route_counts=summary)
