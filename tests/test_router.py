from zippedtext.online_manifest import AnalysisManifest
from zippedtext.router import route_segments
from zippedtext.segment import split_text_segments
from zippedtext.term_dictionary import build_structured_phrase_table


def test_split_text_segments_splits_paragraphs_and_long_prose():
    text = (
        "第一段包含很多很多文字。" * 30
        + "\n\n"
        + "- 条目一\n- 条目二\n- 条目三"
    )
    segments = split_text_segments(text, AnalysisManifest(), max_chars=80)
    assert len(segments) >= 3
    assert segments[-1].kind == "list"
    assert all(segment.char_count > 0 for segment in segments)


def test_route_segments_prefers_phrase_when_repetition_is_high():
    text = "压缩算法在线模式压缩算法在线模式压缩算法在线模式。"
    analysis = AnalysisManifest.from_api_payload(
        {"phrase_dictionary": ["压缩算法", "在线模式"]},
        len(text),
    )
    phrase_table = build_structured_phrase_table(text, analysis)
    segments = split_text_segments(text, analysis, max_chars=200)
    summary = route_segments(
        text=text,
        segments=segments,
        phrase_set=frozenset(phrase_table.phrases),
        priors=None,
        max_order=4,
    )
    routes = [segment.route for segment in summary.routed_segments]
    assert "phrase" in routes
