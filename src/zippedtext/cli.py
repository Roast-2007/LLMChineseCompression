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
@click.option(
    "--sub-mode",
    type=click.Choice(["char", "token"]),
    default="char",
    help="Online sub-mode: char (default) or token (experimental)",
)
@click.option("--model", default="deepseek-chat", help="DeepSeek model name")
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", default=None, help="DeepSeek API key")
@click.option("--base-url", default="https://api.deepseek.com", help="DeepSeek API base URL")
def c(
    input_file: str,
    output: str | None,
    mode: str,
    sub_mode: str,
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

    if mode == "online" and not api_key:
        click.echo("Error: online mode requires --api-key or DEEPSEEK_API_KEY", err=True)
        sys.exit(1)

    api_client = _make_api_client(api_key, model, base_url)

    original_size = len(text.encode("utf-8"))
    click.echo(f"Compressing {input_file} ({original_size:,} bytes, {len(text):,} chars)")
    click.echo(f"Mode: {mode} | Sub-mode: {sub_mode} | Model: {model}")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total
        click.echo(f"\r  [{pct:3d}%] {current}/{total} chars", nl=False)

    data = compress(
        text,
        mode=mode,
        api_client=api_client,
        model_name=model,
        sub_mode=sub_mode,
        on_progress=on_progress,
    )
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

    # Check if this is an online file and warn early about API key
    header, _, _ = read_file(data)
    if header.mode == MODE_ONLINE and not api_key:
        click.echo(
            "Error: this file was compressed in online mode. "
            "Provide --api-key or DEEPSEEK_API_KEY to decompress.",
            err=True,
        )
        sys.exit(1)

    api_client = _make_api_client(api_key, "deepseek-chat", base_url)

    mode_str = "online" if header.mode == MODE_ONLINE else "offline"
    click.echo(f"Decompressing {input_file} ({len(data):,} bytes, mode: {mode_str})")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total
        click.echo(f"\r  [{pct:3d}%] {current}/{total} tokens", nl=False)

    text = decompress(data, api_client=api_client, on_progress=on_progress)
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

    # Online mode extra info
    if header.mode == MODE_ONLINE and model_data:
        from .compressor import _unpack_online_model_data, SUB_MODE_TOKEN
        sub_mode, api_model = _unpack_online_model_data(model_data)
        sub_str = "token" if sub_mode == SUB_MODE_TOKEN else "char"
        click.echo(f"  Sub-mode:       {sub_str}")
        if api_model:
            click.echo(f"  API model:      {api_model}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", default=None)
@click.option("--base-url", default="https://api.deepseek.com")
@click.option("--model", default="deepseek-chat")
def bench(input_file: str, api_key: str | None, base_url: str, model: str) -> None:
    """Benchmark compression on a text file, comparing modes and baselines."""
    import gzip

    text = _read_text(input_file)
    text_bytes = text.encode("utf-8")
    original = len(text_bytes)
    click.echo(f"Benchmarking: {input_file}")
    click.echo(f"  Text: {len(text):,} chars, {original:,} bytes (UTF-8)\n")

    # Baselines
    gz = gzip.compress(text_bytes, compresslevel=9)
    zs = __import__("zstandard").ZstdCompressor(level=19).compress(text_bytes)
    click.echo(f"  {'Method':<30} {'Size':>8} {'Ratio':>8}")
    click.echo(f"  {'─'*30} {'─'*8} {'─'*8}")
    click.echo(f"  {'Raw UTF-8':<30} {original:>8,} {'1.000':>8}")
    click.echo(f"  {'gzip -9':<30} {len(gz):>8,} {len(gz)/original:>8.3f}")
    click.echo(f"  {'zstd -19':<30} {len(zs):>8,} {len(zs)/original:>8.3f}")

    # Offline mode (no API)
    t0 = time.perf_counter()
    offline = compress(text, mode="offline")
    t_offline = time.perf_counter() - t0
    click.echo(f"  {'zippedtext offline':<30} {len(offline):>8,} {len(offline)/original:>8.3f}  ({t_offline:.2f}s)")

    # Online mode with API (if available)
    api_client = _make_api_client(api_key, model, base_url)
    if api_client:
        # Character-level online
        t0 = time.perf_counter()
        online_char = compress(text, mode="online", api_client=api_client, model_name=model, sub_mode="char")
        t_char = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (char)':<30} {len(online_char):>8,} {len(online_char)/original:>8.3f}  ({t_char:.2f}s)")

        # Token-level online
        t0 = time.perf_counter()
        online_token = compress(text, mode="online", api_client=api_client, model_name=model, sub_mode="token")
        t_token = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (token)':<30} {len(online_token):>8,} {len(online_token)/original:>8.3f}  ({t_token:.2f}s)")
    else:
        click.echo(f"  {'zippedtext online (char)':<30} {'(no API key)':>8}")
        click.echo(f"  {'zippedtext online (token)':<30} {'(no API key)':>8}")


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def _make_api_client(api_key: str | None, model: str, base_url: str):
    if not api_key:
        return None
    from .api_client import DeepSeekClient
    return DeepSeekClient(api_key=api_key, model=model, base_url=base_url)
