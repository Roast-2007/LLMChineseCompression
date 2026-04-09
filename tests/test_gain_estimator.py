from zippedtext.gain_estimator import GainEstimate, GainEstimatorConfig, choose_best_route
from zippedtext.segment import TextSegment


def test_choose_best_route_prefers_non_literal_when_gain_is_large():
    segment = TextSegment(start=0, end=12, kind="prose")
    config = GainEstimatorConfig(min_gain_bytes=2)
    literal = GainEstimate(
        route="literal",
        payload=b"12345678",
        payload_bytes=8,
        side_info_bytes=0,
        total_bytes=32,
        estimated_gain_bytes=4,
    )
    phrase = GainEstimate(
        route="phrase",
        payload=b"1234",
        payload_bytes=4,
        side_info_bytes=2,
        total_bytes=28,
        estimated_gain_bytes=8,
    )
    decision = choose_best_route(segment, original_bytes=36, candidates=(literal, phrase), config=config)
    assert decision.chosen.route == "phrase"


def test_choose_best_route_falls_back_when_gain_is_too_small():
    segment = TextSegment(start=0, end=8, kind="prose")
    config = GainEstimatorConfig(min_gain_bytes=3)
    literal = GainEstimate(
        route="literal",
        payload=b"123456",
        payload_bytes=6,
        side_info_bytes=0,
        total_bytes=30,
        estimated_gain_bytes=2,
    )
    template = GainEstimate(
        route="template",
        payload=b"12",
        payload_bytes=2,
        side_info_bytes=1,
        total_bytes=28,
        estimated_gain_bytes=4,
    )
    decision = choose_best_route(segment, original_bytes=32, candidates=(literal, template), config=config)
    assert decision.chosen.route == "literal"
    assert decision.chosen.fallback_reason == "side-info cost too high"


def test_choose_best_route_preserves_specific_literal_reason():
    segment = TextSegment(start=0, end=8, kind="config")
    config = GainEstimatorConfig(min_gain_bytes=4)
    literal = GainEstimate(
        route="literal",
        payload=b"123456",
        payload_bytes=6,
        side_info_bytes=0,
        total_bytes=30,
        estimated_gain_bytes=2,
        fallback_reason="template below threshold",
    )
    template = GainEstimate(
        route="template",
        payload=b"12",
        payload_bytes=2,
        side_info_bytes=1,
        total_bytes=29,
        estimated_gain_bytes=3,
    )
    decision = choose_best_route(segment, original_bytes=32, candidates=(literal, template), config=config)
    assert decision.chosen.route == "literal"
    assert decision.chosen.fallback_reason == "template below threshold"
