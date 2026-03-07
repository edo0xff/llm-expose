"""CLI entry point for llm-expose using Typer and Rich."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from llm_expose.config import (
    ExposureConfig,
    ProviderConfig,
    TelegramClientConfig,
    list_configs,
    load_config,
    save_config,
)
from llm_expose.core.orchestrator import Orchestrator
from llm_expose.providers.litellm_provider import LiteLLMProvider
from llm_expose.clients.telegram import TelegramClient

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="llm-expose",
    help="Expose LLMs through messaging clients (Telegram, and more).",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# ---------------------------------------------------------------------------
# ASCII art / branding
# ---------------------------------------------------------------------------

_BANNER = r"""
 _    _     __  __   _____                             
| |  | |   |  \/  | |  ___|                            
| |  | |   | \  / | | |__  __  __ _ __    ___   ___  ___ 
| |  | |   | |\/| | |  __| \ \/ /| '_ \  / _ \ / __|/ _ \
| |__| |   | |  | | | |___  >  < | |_) || (_) |\__ \  __/
 \____/    |_|  |_| |_____/ /_/\_\| .__/  \___/ |___/\___|
                                   | |                    
                                   |_|                    
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Display the welcome banner."""
    console.print(Panel(Text(_BANNER, style="bold cyan", justify="center"), border_style="cyan"))


def _select_from_list(prompt: str, options: list[str]) -> str:
    """Ask the user to pick one option from a numbered list.

    Args:
        prompt: Question text shown before the list.
        options: Available choices.

    Returns:
        The selected option string.
    """
    console.print(f"\n[bold]{prompt}[/bold]")
    for idx, option in enumerate(options, start=1):
        console.print(f"  [cyan]{idx})[/cyan] {option}")
    while True:
        raw = Prompt.ask("  Enter number")
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1]
        console.print("[red]  Invalid selection. Please enter a number from the list.[/red]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def create() -> None:
    """Interactively create a new LLM exposure configuration."""
    _print_banner()
    console.print("\n[bold green]Welcome to llm-expose![/bold green]\n")
    console.print("Let's set up a new LLM exposure. Answer the prompts below.\n")

    # ---- Exposure name ------------------------------------------------
    name = Prompt.ask("[bold]Give this exposure a name[/bold]")
    name = name.strip()
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    # ---- Provider type ------------------------------------------------
    provider_type = _select_from_list(
        "Select LLM provider type:",
        ["Local (LM Studio / Ollama / vLLM / OpenAI-compatible)", "Online"],
    )

    if provider_type.startswith("Online"):
        online_provider = _select_from_list(
            "Select online provider:",
            ["openai", "anthropic", "google", "cohere", "mistral", "other"],
        )
        if online_provider == "other":
            online_provider = Prompt.ask("  Enter provider name (as recognised by LiteLLM)")
        provider_name = online_provider.lower().strip()
        base_url: Optional[str] = None
    else:
        provider_name = "local"
        base_url = Prompt.ask(
            "  Base URL of your local server",
            default="http://localhost:1234/v1",
        )

    model = Prompt.ask(f"  Model name for [cyan]{provider_name}[/cyan]")

    api_key: Optional[str] = None
    if provider_name != "local":
        raw_key = Prompt.ask("  API key", password=True, default="")
        api_key = raw_key.strip() or None

    temperature_raw = Prompt.ask("  Temperature", default="0.7")
    try:
        temperature = float(temperature_raw)
    except ValueError:
        console.print("[yellow]  Invalid temperature, using 0.7.[/yellow]")
        temperature = 0.7

    max_tokens_raw = Prompt.ask("  Max tokens", default="2048")
    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        console.print("[yellow]  Invalid max_tokens, using 2048.[/yellow]")
        max_tokens = 2048

    provider_cfg = ProviderConfig(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ---- Client -------------------------------------------------------
    _select_from_list(
        "Select messaging client:",
        ["Telegram"],
        # TODO: Add Discord, Slack once implemented
    )
    bot_token = Prompt.ask("  Telegram bot token (from [link=https://t.me/BotFather]@BotFather[/link])", password=True)
    client_cfg = TelegramClientConfig(bot_token=bot_token.strip())

    # ---- Save ---------------------------------------------------------
    config = ExposureConfig(name=name, provider=provider_cfg, client=client_cfg)
    try:
        saved_path = save_config(config)
    except Exception as exc:
        console.print(f"[red]Failed to save configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Configuration '{name}' saved to {saved_path}[/bold green]\n"
    )

    # ---- Optionally start now -----------------------------------------
    if Confirm.ask("Start the service now?"):
        _start_service(config)


@app.command("list")
def list_command() -> None:
    """List all saved exposure configurations."""
    names = list_configs()
    if not names:
        console.print("[yellow]No configurations found. Run [bold]llm-expose create[/bold] to create one.[/yellow]")
        return

    table = Table(title="Saved Exposures", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="blue")
    table.add_column("Client", style="magenta")

    for idx, name in enumerate(names, start=1):
        try:
            cfg = load_config(name)
            table.add_row(
                str(idx),
                cfg.name,
                cfg.provider.provider_name,
                cfg.provider.model,
                cfg.client.client_type,
            )
        except Exception:
            table.add_row(str(idx), name, "[red]error[/red]", "-", "-")

    console.print(table)


@app.command()
def load(
    name: str = typer.Argument(..., help="Name of the saved exposure to load"),
) -> None:
    """Load a saved configuration and start the service."""
    try:
        config = load_config(name)
    except FileNotFoundError:
        console.print(f"[red]No configuration named '{name}' found.[/red]")
        console.print("Run [bold]llm-expose list[/bold] to see available configurations.")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to load configuration '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Display summary
    _print_config_summary(config)

    if not Confirm.ask("\nStart the service?"):
        raise typer.Exit()

    _start_service(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_config_summary(config: ExposureConfig) -> None:
    """Render a rich summary table for *config*."""
    table = Table(title=f"Configuration: {config.name}", border_style="cyan", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Provider", config.provider.provider_name)
    table.add_row("Model", config.provider.model)
    table.add_row("Temperature", str(config.provider.temperature))
    table.add_row("Max tokens", str(config.provider.max_tokens))
    if config.provider.base_url:
        table.add_row("Base URL", config.provider.base_url)
    table.add_row("Client", config.client.client_type)

    console.print(table)


def _start_service(config: ExposureConfig) -> None:
    """Build the provider + client + orchestrator and run the event loop.

    Args:
        config: Fully validated :class:`~llm_expose.config.models.ExposureConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    provider = LiteLLMProvider(config.provider)
    client = TelegramClient(config.client, handler=_placeholder_handler)
    orchestrator = Orchestrator(config=config, provider=provider, client=client)

    console.print(
        f"\n[bold green]🚀 Starting service '{config.name}'…[/bold green]\n"
        "Press [bold]Ctrl+C[/bold] to stop.\n"
    )

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Service stopped by user.[/yellow]")


async def _placeholder_handler(message: str) -> str:  # pragma: no cover
    """Placeholder; the orchestrator replaces this before the first call."""
    return message


# ---------------------------------------------------------------------------
# CLI package init
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by ``llm-expose`` console script."""
    app()


if __name__ == "__main__":
    main()
