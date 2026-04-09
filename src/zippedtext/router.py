from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .encoder import encode, encode_with_phrases
from .gain_estimator import GainEstimate, GainEstimatorConfig, choose_best_route, estimate_total_bytes
from .online_manifest import ROUTE_LITERAL, ROUTE_PHRASE, ROUTE_TEMPLATE
from .segment import TextSegment
from .template_codec import (
    TemplateCatalog,
    detect_template,
    encode_template_segment,
    template_confidence_threshold,
    template_kind_is_hinted,
    template_route_allowed,
)

PHRASE_ENABLED_KINDS = frozenset({"prose", "list", "table", "config", "mixed"})


@dataclass(frozen=True)
class RoutedSegment:
    segment: TextSegment
    route: str
    payload: bytes
    original_bytes: int
    encoded_bytes: int
    estimated_gain_bytes: int
    residual_bytes: int = 0
    typed_slot_count: int = 0
    typed_template_count: int = 0
    fallback_reason: str = ""


@dataclass(frozen=True)
class RouteSummary:
    routed_segments: tuple[RoutedSegment, ...]
    route_counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]
    estimated_gain_bytes: int
    literal_payload_bytes: int
    residual_bytes: int
    template_hit_count: int
    typed_slot_count: int
    typed_template_count: int
    template_family_counts: tuple[tuple[str, int], ...]


def route_segments(
    text: str,
    segments: tuple[TextSegment, ...],
    phrase_set: frozenset[str],
    priors: dict[str, float] | None,
    max_order: int,
    template_catalog: TemplateCatalog | None = None,
    analysis=None,
    gain_config: GainEstimatorConfig | None = None,
    phrase_section_cost: int = 0,
    template_section_cost: int = 0,
) -> RouteSummary:
    config = gain_config or GainEstimatorConfig()
    routed: list[RoutedSegment] = []
    route_counts: dict[str, int] = {
        ROUTE_LITERAL: 0,
        ROUTE_PHRASE: 0,
        ROUTE_TEMPLATE: 0,
    }
    reason_counts: dict[str, int] = {}
    literal_payload_bytes = 0
    estimated_gain_bytes = 0
    residual_bytes = 0
    typed_slot_count = 0
    typed_template_count = 0
    template_family_counter: Counter[str] = Counter()

    phrase_amortized_cost = _amortize_cost(phrase_section_cost, len(segments))
    template_index_map = {
        entry: index
        for index, entry in enumerate(template_catalog.entries)
    } if template_catalog else {}
    family_counts = Counter(
        f"{match.template_kind}:{match.skeleton}"
        for segment in segments
        for match in [detect_template(text[segment.start:segment.end], analysis)]
        if match is not None and (match.template_kind, match.skeleton) in template_index_map
    )

    for segment in segments:
        segment_text = text[segment.start:segment.end]
        original_bytes = len(segment_text.encode("utf-8"))
        literal_reason = ""
        literal_payload = encode(
            segment_text,
            priors=priors,
            max_order=max_order,
        )
        literal_payload_bytes += len(literal_payload)
        candidates = [
            GainEstimate(
                route=ROUTE_LITERAL,
                payload=literal_payload,
                payload_bytes=len(literal_payload),
                side_info_bytes=0,
                total_bytes=estimate_total_bytes(len(literal_payload), 0, config),
                estimated_gain_bytes=max(original_bytes - len(literal_payload), 0),
                fallback_reason=literal_reason,
            )
        ]

        if phrase_set and segment.kind in PHRASE_ENABLED_KINDS:
            phrase_payload = encode_with_phrases(
                segment_text,
                phrase_set,
                priors=priors,
                max_order=max_order,
            )
            phrase_side_info = phrase_amortized_cost + config.route_switch_penalty_bytes
            phrase_total = estimate_total_bytes(len(phrase_payload), phrase_side_info, config)
            candidates.append(
                GainEstimate(
                    route=ROUTE_PHRASE,
                    payload=phrase_payload,
                    payload_bytes=len(phrase_payload),
                    side_info_bytes=phrase_side_info,
                    total_bytes=phrase_total,
                    estimated_gain_bytes=max(original_bytes - phrase_total, 0),
                )
            )

        chosen_match = None
        if template_catalog and template_route_allowed(segment.kind):
            match = detect_template(segment_text, analysis)
            if match:
                entry = (match.template_kind, match.skeleton)
                template_index = template_index_map.get(entry)
                if template_index is None:
                    literal_reason = "template no catalog reuse"
                else:
                    threshold = template_confidence_threshold(match, analysis)
                    if template_kind_is_hinted(match.template_kind, analysis):
                        threshold = max(threshold - 0.03, 0.6)
                    if match.confidence < threshold:
                        literal_reason = "template below threshold"
                    else:
                        template_payload = encode_template_segment(
                            segment_text,
                            match,
                            template_index,
                            phrase_set,
                            priors,
                            max_order,
                        )
                        family_key = f"{match.template_kind}:{match.skeleton}"
                        template_side_info = _family_amortized_cost(
                            template_section_cost,
                            family_counts.get(family_key, 1),
                        ) + config.route_switch_penalty_bytes
                        template_total = estimate_total_bytes(
                            template_payload.encoded_bytes,
                            template_side_info,
                            config,
                        )
                        candidates.append(
                            GainEstimate(
                                route=ROUTE_TEMPLATE,
                                payload=template_payload.payload,
                                payload_bytes=template_payload.encoded_bytes,
                                side_info_bytes=template_side_info,
                                total_bytes=template_total,
                                estimated_gain_bytes=max(original_bytes - template_total, 0),
                                residual_bytes=template_payload.residual_bytes,
                            )
                        )
                        chosen_match = match

        if literal_reason:
            candidates[0] = GainEstimate(
                route=candidates[0].route,
                payload=candidates[0].payload,
                payload_bytes=candidates[0].payload_bytes,
                side_info_bytes=candidates[0].side_info_bytes,
                total_bytes=candidates[0].total_bytes,
                estimated_gain_bytes=candidates[0].estimated_gain_bytes,
                fallback_reason=literal_reason,
                residual_bytes=candidates[0].residual_bytes,
            )

        decision = choose_best_route(segment, original_bytes, tuple(candidates), config).chosen
        route_counts[decision.route] = route_counts.get(decision.route, 0) + 1
        estimated_gain_bytes += decision.estimated_gain_bytes
        residual_bytes += decision.residual_bytes
        segment_typed_slots = 0
        segment_typed_templates = 0
        if decision.route == ROUTE_TEMPLATE and chosen_match is not None:
            segment_typed_slots = sum(1 for slot_type in chosen_match.slot_types if slot_type != "string")
            if segment_typed_slots:
                segment_typed_templates = 1
                typed_slot_count += segment_typed_slots
                typed_template_count += 1
                template_family_counter[f"{chosen_match.template_kind}:{chosen_match.skeleton}"] += 1
        if decision.route == ROUTE_LITERAL and decision.fallback_reason:
            reason_counts[decision.fallback_reason] = reason_counts.get(decision.fallback_reason, 0) + 1
        routed.append(
            RoutedSegment(
                segment=segment,
                route=decision.route,
                payload=decision.payload,
                original_bytes=original_bytes,
                encoded_bytes=decision.payload_bytes,
                estimated_gain_bytes=decision.estimated_gain_bytes,
                residual_bytes=decision.residual_bytes,
                typed_slot_count=segment_typed_slots,
                typed_template_count=segment_typed_templates,
                fallback_reason=decision.fallback_reason,
            )
        )

    summary = tuple((route, count) for route, count in sorted(route_counts.items()) if count > 0)
    reasons = tuple(sorted(reason_counts.items(), key=lambda item: item[0]))
    template_families = tuple(sorted(template_family_counter.items(), key=lambda item: item[0]))
    return RouteSummary(
        routed_segments=tuple(routed),
        route_counts=summary,
        reason_counts=reasons,
        estimated_gain_bytes=estimated_gain_bytes,
        literal_payload_bytes=literal_payload_bytes,
        residual_bytes=residual_bytes,
        template_hit_count=route_counts.get(ROUTE_TEMPLATE, 0),
        typed_slot_count=typed_slot_count,
        typed_template_count=typed_template_count,
        template_family_counts=template_families,
    )


def _amortize_cost(total_cost: int, segment_count: int) -> int:
    if total_cost <= 0 or segment_count <= 0:
        return 0
    return max(total_cost // segment_count, 1)


def _family_amortized_cost(total_cost: int, family_count: int) -> int:
    if total_cost <= 0 or family_count <= 0:
        return 0
    return max(total_cost // family_count, 1)
