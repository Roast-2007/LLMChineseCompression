"""CLI ``config`` subcommand group — interactive setup, model listing, etc."""

from __future__ import annotations

import sys

import click

from .config import (
    AppConfig,
    load_config,
    mask_key,
    resolve_config,
    save_config,
)
from .provider import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    KNOWN_PROVIDERS,
    get_provider,
    list_provider_names,
)


@click.group()
def config() -> None:
    """Manage ZippedText configuration (API provider, model, etc.)."""


@config.command()
def init() -> None:
    """Interactive guided configuration setup."""
    click.echo("=== ZippedText 配置向导 ===\n")

    # Step 1: Choose provider
    click.echo("可用的 API 平台：")
    names = list_provider_names()
    for i, name in enumerate(names, 1):
        info = KNOWN_PROVIDERS[name]
        click.echo(f"  {i}. {info.display_name} ({info.default_base_url})")
    click.echo(f"  {len(names) + 1}. 自定义 (Custom OpenAI-compatible API)")

    choice = click.prompt(
        "\n选择平台编号",
        type=click.IntRange(1, len(names) + 1),
        default=1,
    )

    if choice <= len(names):
        provider_name = names[choice - 1]
        provider_info = KNOWN_PROVIDERS[provider_name]
        base_url = provider_info.default_base_url
        click.echo(f"\n已选择: {provider_info.display_name}")
        click.echo(f"API 地址: {base_url}")
    else:
        provider_name = "custom"
        base_url = click.prompt("\n请输入 API Base URL", type=str)

    # Step 2: API key
    api_key = click.prompt(
        "\n请输入 API Key",
        type=str,
        hide_input=True,
    )
    if not api_key.strip():
        click.echo("警告：未输入 API Key，在线模式将不可用", err=True)
        api_key = ""

    # Step 3: Try to fetch models
    model = DEFAULT_MODEL
    if api_key:
        click.echo("\n正在获取可用模型列表...")
        try:
            from .api_client import ApiClient
            client = ApiClient(api_key=api_key, model="", base_url=base_url)
            models = client.list_models()
            if models:
                click.echo(f"\n找到 {len(models)} 个可用模型：")
                display_models = models[:20]
                for i, m in enumerate(display_models, 1):
                    click.echo(f"  {i}. {m}")
                if len(models) > 20:
                    click.echo(f"  ... 共 {len(models)} 个模型")

                idx = click.prompt(
                    "\n选择模型编号（或输入 0 手动填写）",
                    type=click.IntRange(0, len(display_models)),
                    default=1,
                )
                if idx > 0:
                    model = display_models[idx - 1]
                else:
                    model = click.prompt("请输入模型名称", type=str, default=DEFAULT_MODEL)
            else:
                click.echo("未获取到模型列表，将使用默认模型")
                model = click.prompt("请输入模型名称", type=str, default=DEFAULT_MODEL)
        except Exception as e:
            click.echo(f"获取模型列表失败: {e}", err=True)
            model = click.prompt("请手动输入模型名称", type=str, default=DEFAULT_MODEL)
    else:
        model = click.prompt("\n请输入模型名称", type=str, default=DEFAULT_MODEL)

    # Save
    cfg = AppConfig(
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    save_config(cfg)
    click.echo(f"\n配置已保存到 {save_config.__module__}")
    click.echo("配置概览：")
    _print_config(cfg)


@config.command()
def show() -> None:
    """Display the current effective configuration."""
    cfg = resolve_config()
    click.echo("当前生效配置：")
    _print_config(cfg)


@config.command("set")
@click.argument("key")
@click.argument("value")
def set_value(key: str, value: str) -> None:
    """Set a single configuration value.

    Supported keys: provider, base_url, api_key, model
    """
    valid_keys = {"provider", "base_url", "api_key", "model"}
    if key not in valid_keys:
        click.echo(f"无效的配置项: {key}", err=True)
        click.echo(f"可用配置项: {', '.join(sorted(valid_keys))}", err=True)
        sys.exit(1)

    cfg = load_config()
    updates = {key: value}
    new_cfg = AppConfig(
        provider=updates.get("provider", cfg.provider),
        base_url=updates.get("base_url", cfg.base_url),
        api_key=updates.get("api_key", cfg.api_key),
        model=updates.get("model", cfg.model),
    )
    save_config(new_cfg)
    click.echo(f"已更新 {key} = {mask_key(value) if key == 'api_key' else value}")


@config.command()
def models() -> None:
    """Fetch and display available models from the configured provider."""
    cfg = resolve_config()
    if not cfg.api_key:
        click.echo("错误：未配置 API Key，请先运行 zippedtext config init", err=True)
        sys.exit(1)

    click.echo(f"正在从 {cfg.base_url} 获取模型列表...")
    try:
        from .api_client import ApiClient
        client = ApiClient(api_key=cfg.api_key, model="", base_url=cfg.base_url)
        model_list = client.list_models()
    except Exception as e:
        click.echo(f"获取失败: {e}", err=True)
        sys.exit(1)

    if not model_list:
        click.echo("未找到可用模型")
        return

    click.echo(f"\n可用模型（共 {len(model_list)} 个）：")
    for i, m in enumerate(model_list, 1):
        marker = " ← 当前" if m == cfg.model else ""
        click.echo(f"  {i}. {m}{marker}")

    if click.confirm("\n是否切换模型?", default=False):
        idx = click.prompt(
            "选择模型编号",
            type=click.IntRange(1, len(model_list)),
        )
        new_model = model_list[idx - 1]
        new_cfg = AppConfig(
            provider=cfg.provider,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=new_model,
        )
        save_config(new_cfg)
        click.echo(f"已切换模型: {new_model}")


def _print_config(cfg: AppConfig) -> None:
    """Pretty-print a config."""
    provider_info = get_provider(cfg.provider)
    provider_display = provider_info.display_name if provider_info else cfg.provider
    click.echo(f"  平台:    {provider_display}")
    click.echo(f"  API 地址: {cfg.base_url}")
    click.echo(f"  API Key:  {mask_key(cfg.api_key) if cfg.api_key else '(未设置)'}")
    click.echo(f"  模型:     {cfg.model}")
