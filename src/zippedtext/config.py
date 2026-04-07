"""Persistent configuration for API provider, key, model, etc.

Config file: ``~/.zippedtext/config.json``

Resolution priority (highest first):
  1. CLI flags (--api-key, --base-url, --model)
  2. Environment variables (ZIPPEDTEXT_API_KEY, provider-specific)
  3. Config file
  4. Built-in defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .provider import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    KNOWN_PROVIDERS,
    get_provider,
)

CONFIG_DIR = Path.home() / ".zippedtext"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass(frozen=True)
class AppConfig:
    """Resolved application configuration."""
    provider: str = DEFAULT_PROVIDER
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = DEFAULT_MODEL


def load_config() -> AppConfig:
    """Load config from disk, returning defaults if file is absent."""
    if not CONFIG_FILE.exists():
        return AppConfig()
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return AppConfig(
            provider=raw.get("provider", DEFAULT_PROVIDER),
            base_url=raw.get("base_url", "https://api.deepseek.com"),
            api_key=raw.get("api_key", ""),
            model=raw.get("model", DEFAULT_MODEL),
        )
    except (json.JSONDecodeError, OSError):
        return AppConfig()


def save_config(config: AppConfig) -> None:
    """Write config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_config(
    cli_api_key: str | None = None,
    cli_base_url: str | None = None,
    cli_model: str | None = None,
) -> AppConfig:
    """Merge CLI flags, env vars, and config file into a single config.

    Priority: CLI > env > file > defaults.
    """
    file_cfg = load_config()

    # Determine provider info for env-var lookup
    provider_info = get_provider(file_cfg.provider)
    env_keys = ["ZIPPEDTEXT_API_KEY"]
    if provider_info:
        env_keys.append(provider_info.env_key_name)
    # Also check legacy DEEPSEEK_API_KEY
    env_keys.append("DEEPSEEK_API_KEY")

    # Resolve API key: CLI > env > file
    api_key = cli_api_key or ""
    if not api_key:
        for ek in env_keys:
            val = os.environ.get(ek, "")
            if val:
                api_key = val
                break
    if not api_key:
        api_key = file_cfg.api_key

    return AppConfig(
        provider=file_cfg.provider,
        base_url=cli_base_url or file_cfg.base_url,
        api_key=api_key,
        model=cli_model or file_cfg.model,
    )


def mask_key(key: str) -> str:
    """Mask an API key for display: show first 6 and last 4 chars."""
    if len(key) <= 12:
        return "***"
    return key[:6] + "..." + key[-4:]
