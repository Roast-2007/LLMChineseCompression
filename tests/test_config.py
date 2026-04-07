"""Tests for configuration system."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from zippedtext.config import (
    AppConfig,
    load_config,
    mask_key,
    resolve_config,
    save_config,
)
from zippedtext.provider import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    KNOWN_PROVIDERS,
    get_provider,
    list_provider_names,
)


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.provider == DEFAULT_PROVIDER
        assert cfg.model == DEFAULT_MODEL
        assert cfg.api_key == ""

    def test_frozen(self):
        cfg = AppConfig()
        with pytest.raises(AttributeError):
            cfg.model = "new"  # type: ignore[misc]


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        cfg = AppConfig(
            provider="siliconflow",
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-test-123",
            model="Qwen/Qwen2.5-7B",
        )
        cfg_file = tmp_path / "config.json"
        with patch("zippedtext.config.CONFIG_FILE", cfg_file), \
             patch("zippedtext.config.CONFIG_DIR", tmp_path):
            save_config(cfg)
            loaded = load_config()

        assert loaded.provider == "siliconflow"
        assert loaded.api_key == "sk-test-123"
        assert loaded.model == "Qwen/Qwen2.5-7B"

    def test_load_missing_file(self, tmp_path: Path):
        cfg_file = tmp_path / "nonexistent" / "config.json"
        with patch("zippedtext.config.CONFIG_FILE", cfg_file):
            cfg = load_config()
        assert cfg == AppConfig()

    def test_load_corrupt_json(self, tmp_path: Path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("{bad json", encoding="utf-8")
        with patch("zippedtext.config.CONFIG_FILE", cfg_file):
            cfg = load_config()
        assert cfg == AppConfig()


class TestResolveConfig:
    def test_cli_overrides_all(self, tmp_path: Path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "api_key": "file-key",
            "model": "file-model",
        }), encoding="utf-8")

        with patch("zippedtext.config.CONFIG_FILE", cfg_file), \
             patch.dict(os.environ, {}, clear=True):
            cfg = resolve_config(
                cli_api_key="cli-key",
                cli_base_url="https://custom.api",
                cli_model="cli-model",
            )
        assert cfg.api_key == "cli-key"
        assert cfg.base_url == "https://custom.api"
        assert cfg.model == "cli-model"

    def test_env_overrides_file(self, tmp_path: Path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "api_key": "file-key",
            "model": "deepseek-chat",
        }), encoding="utf-8")

        with patch("zippedtext.config.CONFIG_FILE", cfg_file), \
             patch.dict(os.environ, {"ZIPPEDTEXT_API_KEY": "env-key"}, clear=True):
            cfg = resolve_config()
        assert cfg.api_key == "env-key"

    def test_legacy_deepseek_env(self, tmp_path: Path):
        cfg_file = tmp_path / "nonexist.json"
        with patch("zippedtext.config.CONFIG_FILE", cfg_file), \
             patch.dict(os.environ, {"DEEPSEEK_API_KEY": "legacy-key"}, clear=True):
            cfg = resolve_config()
        assert cfg.api_key == "legacy-key"

    def test_file_fallback(self, tmp_path: Path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "api_key": "saved-key",
            "model": "saved-model",
        }), encoding="utf-8")

        with patch("zippedtext.config.CONFIG_FILE", cfg_file), \
             patch.dict(os.environ, {}, clear=True):
            cfg = resolve_config()
        assert cfg.api_key == "saved-key"
        assert cfg.model == "saved-model"


class TestProviders:
    def test_known_providers(self):
        assert "deepseek" in KNOWN_PROVIDERS
        assert "siliconflow" in KNOWN_PROVIDERS
        assert "openai" in KNOWN_PROVIDERS

    def test_get_provider(self):
        info = get_provider("siliconflow")
        assert info is not None
        assert "siliconflow" in info.default_base_url

    def test_get_unknown(self):
        assert get_provider("nonexistent") is None

    def test_list_names(self):
        names = list_provider_names()
        assert isinstance(names, list)
        assert "deepseek" in names


class TestMaskKey:
    def test_long_key(self):
        assert mask_key("sk-1234567890abcdef") == "sk-123...cdef"

    def test_short_key(self):
        assert mask_key("short") == "***"

    def test_empty(self):
        assert mask_key("") == "***"
