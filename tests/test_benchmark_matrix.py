"""Benchmark matrix: structured test samples with compression ratio comparison."""

import pytest
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "benchmark_samples"


@pytest.mark.benchmark
@pytest.mark.parametrize("sample_file", [
    "api_doc_sample.txt",
    "config_sample.txt",
    "release_notes_sample.txt",
    "faq_sample.txt",
    "bilingual_terms_sample.txt",
    "mixed_prose_sample.txt",
])
def test_benchmark_sample_roundtrips(sample_file):
    """Verify every benchmark sample round-trips correctly."""
    from zippedtext.compressor import compress, decompress

    text = (SAMPLES_DIR / sample_file).read_text(encoding="utf-8")
    data = compress(text, mode="offline")
    assert decompress(data) == text


@pytest.mark.benchmark
def test_benchmark_matrix_all_samples_exist():
    """Verify all expected sample files exist."""
    expected = [
        "api_doc_sample.txt",
        "config_sample.txt",
        "release_notes_sample.txt",
        "faq_sample.txt",
        "bilingual_terms_sample.txt",
        "mixed_prose_sample.txt",
    ]
    for name in expected:
        assert (SAMPLES_DIR / name).exists(), f"Missing sample: {name}"
