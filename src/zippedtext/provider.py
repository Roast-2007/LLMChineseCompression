"""Known API provider presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for a known OpenAI-compatible API provider."""
    name: str
    display_name: str
    default_base_url: str
    env_key_name: str


KNOWN_PROVIDERS: dict[str, ProviderInfo] = {
    "deepseek": ProviderInfo(
        name="deepseek",
        display_name="DeepSeek",
        default_base_url="https://api.deepseek.com",
        env_key_name="DEEPSEEK_API_KEY",
    ),
    "siliconflow": ProviderInfo(
        name="siliconflow",
        display_name="SiliconFlow 硅基流动",
        default_base_url="https://api.siliconflow.cn/v1",
        env_key_name="SILICONFLOW_API_KEY",
    ),
    "openai": ProviderInfo(
        name="openai",
        display_name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        env_key_name="OPENAI_API_KEY",
    ),
}

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"


def get_provider(name: str) -> ProviderInfo | None:
    """Look up a provider by name (case-insensitive)."""
    return KNOWN_PROVIDERS.get(name.lower())


def list_provider_names() -> list[str]:
    """Return sorted list of known provider names."""
    return sorted(KNOWN_PROVIDERS.keys())
