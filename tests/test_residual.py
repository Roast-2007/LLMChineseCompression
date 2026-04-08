from zippedtext.decoder import decode, decode_with_phrases
from zippedtext.predictor.phrases import PhraseTable
from zippedtext.residual import deserialize_string_tuple, encode_residual_segments, serialize_string_tuple


def test_string_tuple_roundtrip():
    payload = serialize_string_tuple(("alpha", "beta"))
    assert deserialize_string_tuple(payload) == ("alpha", "beta")


def test_encode_residual_segments_roundtrip_literal_and_phrase():
    text = "online mode residual online mode residual"
    phrase_set = frozenset(PhraseTable(phrases=("online mode",)).phrases)
    residual = encode_residual_segments(
        text,
        spans=((0, 11), (11, len(text))),
        phrase_set=phrase_set,
        priors=None,
        max_order=4,
    )
    restored = []
    for segment in residual.segments:
        char_count = segment.original_end - segment.original_start
        if segment.route == "phrase":
            restored.append(decode_with_phrases(segment.payload, char_count, phrase_set, priors=None, max_order=4))
        else:
            restored.append(decode(segment.payload, char_count, priors=None, max_order=4))
    assert "".join(restored) == text
