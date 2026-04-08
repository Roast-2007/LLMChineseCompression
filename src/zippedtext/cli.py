"""Command-line interface for zippedtext."""

from __future__ import annotations

import sys
import time

import click

from . import __version__
from .compressor import SUB_MODE_TOKEN, _unpack_online_model_data, compress, decompress
from .config import resolve_config
from .format import (
    MODE_CODEGEN,
    MODE_OFFLINE,
    MODE_ONLINE,
    SECTION_ANALYSIS,
    SECTION_PHRASE_TABLE,
    SECTION_SEGMENTS,
    SECTION_STATS,
    VERSION_V3,
    read_file,
    read_file_v3,
)
from .online_manifest import StructuredOnlineStats


@click.group()
@click.version_option(__version__)
def main() -> None:
    """ZippedText — LLM-enhanced lossless text compression."""


from .cli_config import config as config_group  # noqa: E402

main.add_command(config_group)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output .ztxt file path")
@click.option(
    "--mode",
    type=click.Choice(["online", "offline", "codegen"]),
    default="offline",
    help="Compression mode (default: offline, codegen: experimental)",
)
@click.option(
    "--sub-mode",
    type=click.Choice(["structured", "char", "token"]),
    default="structured",
    help="Online sub-mode: structured (default), char, or token",
)
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--api-key", default=None, help="API key (overrides config/env)")
@click.option("--base-url", default=None, help="API base URL (overrides config)")
@click.option("--max-order", type=click.IntRange(4, 6), default=4, help="PPM context depth (4-6, default: 4)")
@click.option("--no-priors", is_flag=True, default=False, help="Disable Chinese frequency priors")
@click.option("--phrases", is_flag=True, default=False, help="Enable phrase-level encoding (experimental)")
def c(
    input_file: str,
    output: str | None,
    mode: str,
    sub_mode: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    max_order: int,
    no_priors: bool,
    phrases: bool,
) -> None:
    """Compress a text file."""
    output = output or (input_file + ".ztxt")

    text = _read_text(input_file)
    if not text:
        click.echo("Error: input file is empty", err=True)
        sys.exit(1)

    cfg = resolve_config(cli_api_key=api_key, cli_base_url=base_url, cli_model=model)

    if mode in ("online", "codegen") and not cfg.api_key:
        click.echo(
            f"Error: {mode} mode requires an API key.\n"
            "  Set via --api-key, env var, or run: zippedtext config init",
            err=True,
        )
        sys.exit(1)

    api_client = _make_api_client(cfg)

    original_size = len(text.encode("utf-8"))
    click.echo(f"Compressing {input_file} ({original_size:,} bytes, {len(text):,} chars)")
    click.echo(f"Mode: {mode} | Sub-mode: {sub_mode} | Model: {cfg.model}")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total if total else 100
        click.echo(f"\r  [{pct:3d}%] {current}/{total} chars", nl=False)

    data = compress(
        text,
        mode=mode,
        api_client=api_client,
        model_name=cfg.model,
        sub_mode=sub_mode,
        on_progress=on_progress,
        use_priors=not no_priors,
        max_order=max_order,
        use_phrases=phrases,
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
@click.option("--api-key", default=None, help="API key (overrides config/env)")
@click.option("--base-url", default=None, help="API base URL (overrides config)")
def d(
    input_file: str,
    output: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Decompress a .ztxt file."""
    if output is None:
        if input_file.endswith(".ztxt"):
            output = input_file[:-5]
        else:
            output = input_file + ".txt"

    with open(input_file, "rb") as f:
        data = f.read()

    header, model_data, _ = read_file(data)
    cfg = resolve_config(cli_api_key=api_key, cli_base_url=base_url)

    has_cache = header.mode == MODE_ONLINE and header.version == VERSION_V3
    if header.mode == MODE_ONLINE and header.version != VERSION_V3 and model_data:
        _, _, _, _, cc, tc = _unpack_online_model_data(model_data)
        has_cache = cc is not None or tc is not None

    if header.mode == MODE_ONLINE and not cfg.api_key and not has_cache:
        click.echo(
            "Error: this file was compressed in online mode.\n"
            "  Provide --api-key, set env var, or run: zippedtext config init",
            err=True,
        )
        sys.exit(1)

    if header.mode == MODE_ONLINE and header.version != VERSION_V3 and model_data:
        _, file_model, _, _, _, _ = _unpack_online_model_data(model_data)
        if file_model and file_model != cfg.model:
            click.echo(
                f"Warning: file was compressed with model '{file_model}', "
                f"current config uses '{cfg.model}'. Using file's model.",
                err=True,
            )
            cfg = resolve_config(cli_api_key=api_key, cli_base_url=base_url, cli_model=file_model)

    api_client = _make_api_client(cfg)

    mode_map = {MODE_ONLINE: "online", MODE_OFFLINE: "offline", MODE_CODEGEN: "codegen"}
    mode_str = mode_map.get(header.mode, "unknown")
    click.echo(f"Decompressing {input_file} ({len(data):,} bytes, mode: {mode_str})")

    t0 = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        pct = current * 100 // total if total else 100
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
    mode_map = {MODE_ONLINE: "online", MODE_OFFLINE: "offline", MODE_CODEGEN: "codegen"}
    mode_str = mode_map.get(header.mode, f"unknown({header.mode})")

    click.echo(f"File: {input_file}")
    click.echo(f"  Size:           {len(data):,} bytes")
    click.echo(f"  Version:        v{header.version}")
    click.echo(f"  Mode:           {mode_str}")
    click.echo(f"  Max order:      {header.max_order}")
    click.echo(f"  Token count:    {header.token_count:,}")
    click.echo(f"  Original size:  {header.original_bytes:,} bytes")
    click.echo(f"  CRC32:          {header.crc32:#010x}")
    click.echo(f"  Model data:     {header.model_data_len:,} bytes")
    click.echo(f"  Compressed body:{len(body):,} bytes")
    if header.original_bytes > 0:
        ratio = len(data) / header.original_bytes
        click.echo(f"  Ratio:          {ratio:.3f}")

    flags_parts = []
    if header.flags & 0x01:
        flags_parts.append("phrases")
    if header.flags & 0x02:
        flags_parts.append("priors")
    if flags_parts:
        click.echo(f"  Flags:          {', '.join(flags_parts)}")

    if header.mode == MODE_ONLINE and header.version == VERSION_V3:
        _, sections, _ = read_file_v3(data)
        click.echo("  Online path:    structured")
        click.echo(f"  Analysis:       {'yes' if SECTION_ANALYSIS in sections else 'no'}")
        click.echo(f"  Dictionary:     {'yes' if SECTION_PHRASE_TABLE in sections else 'no'}")
        if SECTION_SEGMENTS in sections:
            stats = StructuredOnlineStats.deserialize(sections.get(SECTION_STATS, b""))
            click.echo(f"  Segments:       {stats.segment_count}")
            click.echo(f"  Phrase count:   {stats.phrase_count}")
            if stats.route_counts:
                route_text = ", ".join(f"{route}={count}" for route, count in stats.route_counts)
                click.echo(f"  Routes:         {route_text}")
            click.echo("  Decompress:     API-free")
    elif header.mode == MODE_ONLINE and model_data:
        sub_mode, api_model, chunk_chars, max_tokens, char_cache, tok_cache = _unpack_online_model_data(model_data)
        sub_str = "token" if sub_mode == SUB_MODE_TOKEN else "char"
        click.echo(f"  Online path:    legacy-{sub_str}")
        if api_model:
            click.echo(f"  API model:      {api_model}")
        click.echo(f"  Chunk chars:    {chunk_chars}")
        click.echo(f"  Max tokens:     {max_tokens}")
        has_cache = char_cache is not None or tok_cache is not None
        click.echo(f"  Pred. cache:    {'yes (API-free decompress)' if has_cache else 'no'}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="API base URL")
@click.option("--model", default=None, help="Model name")
def bench(input_file: str, api_key: str | None, base_url: str | None, model: str | None) -> None:
    """Benchmark compression on a text file, comparing modes and baselines."""
    import gzip

    cfg = resolve_config(cli_api_key=api_key, cli_base_url=base_url, cli_model=model)

    text = _read_text(input_file)
    text_bytes = text.encode("utf-8")
    original = len(text_bytes)
    click.echo(f"Benchmarking: {input_file}")
    click.echo(f"  Text: {len(text):,} chars, {original:,} bytes (UTF-8)\n")

    gz = gzip.compress(text_bytes, compresslevel=9)
    zs = __import__("zstandard").ZstdCompressor(level=19).compress(text_bytes)
    click.echo(f"  {'Method':<30} {'Size':>8} {'Ratio':>8}")
    click.echo(f"  {'─'*30} {'─'*8} {'─'*8}")
    click.echo(f"  {'Raw UTF-8':<30} {original:>8,} {'1.000':>8}")
    click.echo(f"  {'gzip -9':<30} {len(gz):>8,} {len(gz)/original:>8.3f}")
    click.echo(f"  {'zstd -19':<30} {len(zs):>8,} {len(zs)/original:>8.3f}")

    t0 = time.perf_counter()
    offline = compress(text, mode="offline")
    t_offline = time.perf_counter() - t0
    click.echo(f"  {'zippedtext offline':<30} {len(offline):>8,} {len(offline)/original:>8.3f}  ({t_offline:.2f}s)")

    api_client = _make_api_client(cfg)
    if api_client:
        t0 = time.perf_counter()
        online_structured = compress(text, mode="online", api_client=api_client, model_name=cfg.model, sub_mode="structured")
        t_structured = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (structured)':<30} {len(online_structured):>8,} {len(online_structured)/original:>8.3f}  ({t_structured:.2f}s)")

        t0 = time.perf_counter()
        online_char = compress(text, mode="online", api_client=api_client, model_name=cfg.model, sub_mode="char")
        t_char = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (legacy char)':<30} {len(online_char):>8,} {len(online_char)/original:>8.3f}  ({t_char:.2f}s)")

        t0 = time.perf_counter()
        online_token = compress(text, mode="online", api_client=api_client, model_name=cfg.model, sub_mode="token")
        t_token = time.perf_counter() - t0
        click.echo(f"  {'zippedtext online (legacy token)':<30} {len(online_token):>8,} {len(online_token)/original:>8.3f}  ({t_token:.2f}s)")
    else:
        click.echo(f"  {'zippedtext online (structured)':<30} {'(no API key)':>8}")
        click.echo(f"  {'zippedtext online (legacy char)':<30} {'(no API key)':>8}")
        click.echo(f"  {'zippedtext online (legacy token)':<30} {'(no API key)':>8}")


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def _make_api_client(cfg):
    """Create an API client from resolved config, or None if no key."""
    if not cfg.api_key:
        return None
    from .api_client import ApiClient
    return ApiClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
