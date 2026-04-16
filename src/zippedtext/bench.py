"""Benchmark matrix runner for structured text samples."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    sample_name: str
    original_bytes: int
    gzip_bytes: int
    zstd_bytes: int
    offline_bytes: int
    structured_available: bool


def run_benchmark_matrix(samples_dir: Path) -> tuple[BenchmarkResult, ...]:
    """Run compression benchmark across all sample files."""
    results: list[BenchmarkResult] = []
    for sample_file in sorted(samples_dir.glob("*.txt")):
        text = sample_file.read_text(encoding="utf-8")
        original_bytes = len(text.encode("utf-8"))

        # gzip -9
        gzip_result = subprocess.run(
            ["gzip", "-9", "-c"],
            input=text.encode("utf-8"),
            capture_output=True,
        )
        gzip_bytes = len(gzip_result.stdout) if gzip_result.returncode == 0 else original_bytes

        # zstd -19
        try:
            zstd_result = subprocess.run(
                ["zstd", "-19", "-c"],
                input=text.encode("utf-8"),
                capture_output=True,
            )
            zstd_bytes = len(zstd_result.stdout) if zstd_result.returncode == 0 else original_bytes
        except FileNotFoundError:
            zstd_bytes = original_bytes

        # zippedtext offline
        from zippedtext.compressor import compress, decompress
        offline_data = compress(text, mode="offline")
        offline_bytes = len(offline_data)

        # Verify roundtrip
        assert decompress(offline_data) == text, f"Roundtrip failed for {sample_file.name}"

        results.append(
            BenchmarkResult(
                sample_name=sample_file.stem,
                original_bytes=original_bytes,
                gzip_bytes=gzip_bytes,
                zstd_bytes=zstd_bytes,
                offline_bytes=offline_bytes,
                structured_available=True,
            )
        )

    return tuple(results)


def print_benchmark_table(results: tuple[BenchmarkResult, ...]) -> None:
    """Print a formatted benchmark table."""
    header = f"{'Sample':<30} {'Original':>10} {'gzip -9':>10} {'zstd -19':>10} {'zippedtext':>12} {'Ratio':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        ratio = r.offline_bytes / r.original_bytes if r.original_bytes > 0 else 0
        print(
            f"{r.sample_name:<30} {r.original_bytes:>10} {r.gzip_bytes:>10} "
            f"{r.zstd_bytes:>10} {r.offline_bytes:>12} {ratio:>8.2%}"
        )
