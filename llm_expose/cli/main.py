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
from telegram.error import Conflict

from llm_expose.config import (
    ExposureConfig,
    ProviderConfig,
    TelegramClientConfig,
    delete_config,
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

# Common models by provider (curated list of popular models)
_MODELS_BY_PROVIDER = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
        "other",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "other",
    ],
    "google": [
        "gemini-2.0-flash-exp",
        "gemini-exp-1206",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "other",
    ],
    "cohere": [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
        "other",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "open-mistral-nemo",
        "other",
    ],
    "github_copilot": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3.5-sonnet",
        "claude-3-5-sonnet-20241022",
        "o1-preview",
        "o1-mini",
        "other",
    ],
}


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
            ["openai", "anthropic", "google", "cohere", "mistral", "github_copilot", "other"],
        )
        if online_provider == "other":
            online_provider = Prompt.ask("  Enter provider name (as recognised by LiteLLM)")
            model = Prompt.ask(f"  Model name for [cyan]{online_provider}[/cyan]")
        else:
            # Show available models for this provider
            available_models = _MODELS_BY_PROVIDER.get(online_provider, ["other"])
            selected_model = _select_from_list(
                f"Select model for {online_provider}:",
                available_models,
            )
            if selected_model == "other":
                model = Prompt.ask(f"  Enter model name for [cyan]{online_provider}[/cyan]")
            else:
                model = selected_model
        
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
    bot_token = ""
    while not bot_token:
        raw_token = Prompt.ask("  Telegram bot token (from [link=https://t.me/BotFather]@BotFather[/link])")
        bot_token = raw_token.strip()
        if not bot_token:
            console.print("[red]  Bot token cannot be empty. Please try again.[/red]")
    client_cfg = TelegramClientConfig(bot_token=bot_token)

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


@app.command()
def edit(
    name: Optional[str] = typer.Argument(None, help="Name of the exposure to edit"),
) -> None:
    """Edit an existing exposure configuration."""
    names = list_configs()
    if not names:
        console.print("[yellow]No configurations found. Run [bold]llm-expose create[/bold] to create one.[/yellow]")
        raise typer.Exit()

    # Select config to edit
    if name is None:
        name = _select_from_list("Select configuration to edit:", names)
    
    # Load existing config
    try:
        config = load_config(name)
    except FileNotFoundError:
        console.print(f"[red]No configuration named '{name}' found.[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to load configuration '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[bold cyan]Editing configuration: {name}[/bold cyan]")
    console.print("[dim]Press Enter to keep current value, or type new value.[/dim]\n")

    # ---- Exposure name ------------------------------------------------
    new_name = Prompt.ask(
        f"[bold]Exposure name[/bold]",
        default=config.name,
    ).strip()
    if not new_name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    # ---- Provider settings --------------------------------------------
    console.print("\n[bold]Provider Settings:[/bold]")
    
    provider_name = Prompt.ask(
        "  Provider name",
        default=config.provider.provider_name,
    ).strip()

    model = Prompt.ask(
        "  Model name",
        default=config.provider.model,
    ).strip()

    base_url_default = config.provider.base_url or ""
    base_url_input = Prompt.ask(
        "  Base URL (leave empty if not needed)",
        default=base_url_default,
    ).strip()
    base_url = base_url_input if base_url_input else None

    # API key (masked input)
    if config.provider.api_key:
        console.print("  [dim]Current API key is set (hidden)[/dim]")
        change_api_key = Confirm.ask("  Change API key?", default=False)
        if change_api_key:
            new_api_key = Prompt.ask("  New API key", default="").strip()
            api_key = new_api_key if new_api_key else None
        else:
            api_key = config.provider.api_key
    else:
        new_api_key = Prompt.ask("  API key (leave empty if not needed)", default="").strip()
        api_key = new_api_key if new_api_key else None

    temperature_str = Prompt.ask(
        "  Temperature",
        default=str(config.provider.temperature),
    )
    try:
        temperature = float(temperature_str)
    except ValueError:
        console.print(f"[yellow]  Invalid temperature, keeping {config.provider.temperature}.[/yellow]")
        temperature = config.provider.temperature

    max_tokens_str = Prompt.ask(
        "  Max tokens",
        default=str(config.provider.max_tokens),
    )
    try:
        max_tokens = int(max_tokens_str)
    except ValueError:
        console.print(f"[yellow]  Invalid max_tokens, keeping {config.provider.max_tokens}.[/yellow]")
        max_tokens = config.provider.max_tokens

    provider_cfg = ProviderConfig(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ---- Client settings ----------------------------------------------
    console.print("\n[bold]Client Settings:[/bold]")
    console.print("  [dim]Current bot token is set (hidden)[/dim]")
    change_token = Confirm.ask("  Change bot token?", default=False)
    
    if change_token:
        bot_token = ""
        while not bot_token:
            raw_token = Prompt.ask("  New Telegram bot token")
            bot_token = raw_token.strip()
            if not bot_token:
                console.print("[red]  Bot token cannot be empty. Please try again.[/red]")
    else:
        bot_token = config.client.bot_token

    client_cfg = TelegramClientConfig(bot_token=bot_token)

    # ---- Save ---------------------------------------------------------
    updated_config = ExposureConfig(name=new_name, provider=provider_cfg, client=client_cfg)
    
    # If name changed, delete old config
    if new_name != name:
        try:
            delete_config(name)
        except Exception as exc:
            console.print(f"[red]Failed to delete old configuration: {exc}[/red]")
            raise typer.Exit(code=1) from exc

    try:
        saved_path = save_config(updated_config)
    except Exception as exc:
        console.print(f"[red]Failed to save configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Configuration '{new_name}' updated successfully![/bold green]\n"
    )


@app.command()
def delete(
    name: Optional[str] = typer.Argument(None, help="Name of the exposure to delete"),
) -> None:
    """Delete a saved exposure configuration."""
    names = list_configs()
    if not names:
        console.print("[yellow]No configurations found.[/yellow]")
        raise typer.Exit()

    # Select config to delete
    if name is None:
        name = _select_from_list("Select configuration to delete:", names)
    
    # Confirm deletion
    if not Confirm.ask(f"[bold red]Are you sure you want to delete '{name}'?[/bold red]", default=False):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        raise typer.Exit()

    try:
        delete_config(name)
        console.print(f"\n[bold green]✓ Configuration '{name}' deleted successfully.[/bold green]\n")
    except FileNotFoundError:
        console.print(f"[red]No configuration named '{name}' found.[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to delete configuration '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc


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
    except Conflict:
        console.print(
            "\n[red]Telegram conflict detected:[/red] another process is already "
            "polling updates for this bot token.\n"
            "Stop other running instances (or webhook consumers) and try again."
        )
        raise typer.Exit(code=1)
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
