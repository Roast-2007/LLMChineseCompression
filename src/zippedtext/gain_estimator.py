from __future__ import annotations

from dataclasses import dataclass

from .segment import TextSegment


@dataclass(frozen=True)
class GainEstimate:
    route: str
    payload: bytes
    payload_bytes: int
    side_info_bytes: int
    total_bytes: int
    estimated_gain_bytes: int
    fallback_reason: str = ""
    residual_bytes: int = 0


@dataclass(frozen=True)
class SegmentRouteDecision:
    segment: TextSegment
    estimates: tuple[GainEstimate, ...]
    chosen: GainEstimate


@dataclass(frozen=True)
class GainEstimatorConfig:
    record_overhead_bytes: int = 24
    route_switch_penalty_bytes: int = 2
    min_gain_bytes: int = 2


def choose_best_route(
    segment: TextSegment,
    original_bytes: int,
    candidates: tuple[GainEstimate, ...],
    config: GainEstimatorConfig,
) -> SegmentRouteDecision:
    if not candidates:
        raise ValueError("route candidates required")
    ordered = sorted(candidates, key=lambda item: (item.total_bytes, item.route))
    literal = next((item for item in ordered if item.route == "literal"), ordered[0])
    best = ordered[0]
    if best.route != literal.route and literal.total_bytes - best.total_bytes < config.min_gain_bytes:
        best = GainEstimate(
            route=literal.route,
            payload=literal.payload,
            payload_bytes=literal.payload_bytes,
            side_info_bytes=literal.side_info_bytes,
            total_bytes=literal.total_bytes,
            estimated_gain_bytes=max(original_bytes - literal.total_bytes, 0),
            fallback_reason="side-info cost too high",
            residual_bytes=literal.residual_bytes,
        )
    elif best.route == "literal" and best.fallback_reason == "":
        best = GainEstimate(
            route=best.route,
            payload=best.payload,
            payload_bytes=best.payload_bytes,
            side_info_bytes=best.side_info_bytes,
            total_bytes=best.total_bytes,
            estimated_gain_bytes=max(original_bytes - best.total_bytes, 0),
            fallback_reason="literal best",
            residual_bytes=best.residual_bytes,
        )
    return SegmentRouteDecision(segment=segment, estimates=tuple(ordered), chosen=best)


def estimate_total_bytes(payload_bytes: int, side_info_bytes: int, config: GainEstimatorConfig) -> int:
    return payload_bytes + side_info_bytes + config.record_overhead_bytes
