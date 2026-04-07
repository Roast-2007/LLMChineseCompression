"""Command-line interface for zippedtext."""

from __future__ import annotations

import os
import sys
import time

import click

from . import __version__
from .compressor import compress, decompress
from .format import (
    MODE_OFFLINE,
    MODE_ONLINE,
    MODEL_DEEPSEEK_CHAT,
    MODEL_DEEPSEEK_REASONER,
    read_file,
)


@click.group()
@click.version_option(__version__)
def main() -> None:
    """ZippedText — LLM-enhanced lossless text compression."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output .ztxt file path")
@click.option(
    "--mode",
    type=click.Choice(["online", "offline"]),
    default="offline",
    help="Compression mode (default: offline)",
)
@click.option("--model", default="deepseek-chat", help="DeepSeek model name")
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", default=None, help="DeepSeek API key")
@click.option("--base-url", default="https://api.deepseek.com", help="DeepSeek API base URL")
def c(
    input_file: str,
    output: str | None,
    mode: str,
    model: str,
    api_key: str | None,
    base_url: str,
) -> None:
    """Compress a text file."""
    output = output or (input_file + ".ztxt")

    text = _read_text(input_file)
    if not text:
        click.echo("Error: input file is empty", err=True)
        sys.exit(1)

    api_client = _make_api_client(api_key, model, base_url)

    original_size = len(text.encode("utf-8"))
    click.echo(f"Compressing {input_file} ({original_size:,} bytes, {len(text):,} chars)")
    click.echo(f"Mode: {mode} | Model: {model}")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total
        click.echo(f"\r  [{pct:3d}%] {current}/{total} chars", nl=False)

    data = compress(text, mode=mode, api_client=api_client, model_name=model, on_progress=on_progress)
    elapsed = time.perf_counter() - t0
    click.echo()

    with open(output, "wb") as f:
        f.write(data)

    compressed_size = len(data)
    ratio = compressed_size / original_size
    click.echo(f"Done in {elapsed:.2f}s")
    click.echo(f"  {original_size:,}B → {compressed_size:,}B (ratio: {ratio:.3f}, saved {(1-ratio)*100:.1f}%)")
    click.echo(f"  Output: {output}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output text file path")
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", default=None, help="DeepSeek API key (for online mode files)")
@click.option("--base-url", default="https://api.deepseek.com", help="DeepSeek API base URL")
def d(
    input_file: str,
    output: str | None,
    api_key: str | None,
    base_url: str,
) -> None:
    """Decompress a .ztxt file."""
    if output is None:
        if input_file.endswith(".ztxt"):
            output = input_file[:-5]
        else:
            output = input_file + ".txt"

    with open(input_file, "rb") as f:
        data = f.read()

    click.echo(f"Decompressing {input_file} ({len(data):,} bytes)")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total
        click.echo(f"\r  [{pct:3d}%] {current}/{total} tokens", nl=False)

    text = decompress(data, on_progress=on_progress)
    elapsed = time.perf_counter() - t0
    click.echo()

    with open(output, "wb") as f:
        f.write(text.encode("utf-8"))

    click.echo(f"Done in {elapsed:.2f}s")
    click.echo(f"  {len(data):,}B → {len(text.encode('utf-8')):,}B ({len(text):,} chars)")
    click.echo(f"  Output: {output}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file: str) -> None:
    """Show info about a .ztxt file."""
    with open(input_file, "rb") as f:
        data = f.read()

    header, model_data, body = read_file(data)
    mode_str = "online" if header.mode == MODE_ONLINE else "offline"
    model_names = {MODEL_DEEPSEEK_CHAT: "deepseek-chat", MODEL_DEEPSEEK_REASONER: "deepseek-reasoner"}
    model_str = model_names.get(header.model_id, f"unknown({header.model_id})")

    click.echo(f"File: {input_file}")
    click.echo(f"  Size:           {len(data):,} bytes")
    click.echo(f"  Mode:           {mode_str}")
    click.echo(f"  Model:          {model_str}")
    click.echo(f"  Token count:    {header.token_count:,}")
    click.echo(f"  Original size:  {header.original_bytes:,} bytes")
    click.echo(f"  CRC32:          {header.crc32:#010x}")
    click.echo(f"  Model data:     {header.model_data_len:,} bytes")
    click.echo(f"  Compressed body:{len(body):,} bytes")
    if header.original_bytes > 0:
        ratio = len(data) / header.original_bytes
        click.echo(f"  Ratio:          {ratio:.3f}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", default=None)
@click.option("--base-url", default="https://api.deepseek.com")
@click.option("--model", default="deepseek-chat")
def bench(input_file: str, api_key: str | None, base_url: str, model: str) -> None:
    """Benchmark compression on a text file, comparing modes and baselines."""
    import gzip
    import zlib

    text = _read_text(input_file)
    text_bytes = text.encode("utf-8")
    original = len(text_bytes)
    click.echo(f"Benchmarking: {input_file}")
    click.echo(f"  Text: {len(text):,} chars, {original:,} bytes (UTF-8)\n")

    # Baselines
    gz = gzip.compress(text_bytes, compresslevel=9)
    zs = __import__("zstandard").ZstdCompressor(level=19).compress(text_bytes)
    click.echo(f"  {'Method':<25} {'Size':>8} {'Ratio':>8}")
    click.echo(f"  {'─'*25} {'─'*8} {'─'*8}")
    click.echo(f"  {'Raw UTF-8':<25} {original:>8,} {'1.000':>8}")
    click.echo(f"  {'gzip -9':<25} {len(gz):>8,} {len(gz)/original:>8.3f}")
    click.echo(f"  {'zstd -19':<25} {len(zs):>8,} {len(zs)/original:>8.3f}")

    # Offline mode (no API)
    t0 = time.perf_counter()
    offline = compress(text, mode="offline")
    t_offline = time.perf_counter() - t0
    click.echo(f"  {'zippedtext offline':<25} {len(offline):>8,} {len(offline)/original:>8.3f}  ({t_offline:.2f}s)")

    # Online mode with API (if available)
    api_client = _make_api_client(api_key, model, base_url)
    if api_client:
        t0 = time.perf_counter()
        online = compress(text, mode="online", api_client=api_client, model_name=model)
        t_online = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (LLM)':<25} {len(online):>8,} {len(online)/original:>8.3f}  ({t_online:.2f}s)")
    else:
        click.echo(f"  {'zippedtext online (LLM)':<25} {'(no API key)':>8}")


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def _make_api_client(api_key: str | None, model: str, base_url: str):
    if not api_key:
        return None
    from .api_client import DeepSeekClient
    return DeepSeekClient(api_key=api_key, model=model, base_url=base_url)
