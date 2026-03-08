"""CLI entry point for llm-expose using Typer and Rich."""

from __future__ import annotations

import asyncio
import logging

from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from telegram.error import Conflict

from litellm import validate_environment, models_by_provider

from llm_expose.config import (
    ProviderConfig,
    TelegramClientConfig,
    delete_channel,
    delete_model,
    list_channels,
    list_models,
    load_channel,
    load_model,
    save_channel,
    save_model,
)
from llm_expose.config.models import ExposureConfig
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

# Subcommands for 'add' and 'delete'
add_app = typer.Typer(help="Add model or channel configurations")
delete_app = typer.Typer(help="Delete model or channel configurations")
list_app = typer.Typer(help="List saved models or channels")

app.add_typer(add_app, name="add")
app.add_typer(delete_app, name="delete")
app.add_typer(list_app, name="list")

console = Console()

# ---------------------------------------------------------------------------
# ASCII art / branding
# ---------------------------------------------------------------------------

_BANNER = r"""
 _     _     __  __   _____                           
| |   | |   |  \/  | | ____|_  ___ __   ___  ___  ___ 
| |   | |   | |\/| | |  _| \ \/ / '_ \ / _ \/ __|/ _ \
| |___| |___| |  | | | |___ >  <| |_) | (_) \__ \  __/
|_____|_____|_|  |_| |_____/_/\_\ .__/ \___/|___/\___|
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
# ADD Commands
# ---------------------------------------------------------------------------


@add_app.command("model")
def add_model() -> None:
    """Add a new model (LLM provider) configuration."""
    _print_banner()
    console.print("\n[bold green]Add a new model configuration[/bold green]\n")

    # ---- Model name ------------------------------------------------
    name = Prompt.ask("[bold]Give this model a name[/bold]")
    name = name.strip()
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    # Check if model already exists
    if name in list_models():
        if not Confirm.ask(f"[yellow]Model '{name}' already exists. Overwrite?[/yellow]", default=False):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    # ---- Provider type ------------------------------------------------
    provider_type = _select_from_list(
        "Select LLM provider type:",
        ["Local (LM Studio / Ollama / vLLM / OpenAI-compatible)", "Online (LiteLLM-supported)"],
    )

    if provider_type.startswith("Online"):
        online_provider = _select_from_list(
            "Select online provider:",
            list(models_by_provider.keys()),
        )
        if online_provider == "other":
            online_provider = Prompt.ask("  Enter provider name (as recognised by LiteLLM visit: https://models.litellm.ai/ for best compatibility)")
            model = Prompt.ask(f"  Model name for [cyan]{online_provider}[/cyan]")
        else:
            # Show available models for this provider
            available_models = list(models_by_provider[online_provider])
            selected_model = _select_from_list(
                f"Select model for {online_provider}:",
                available_models,
            )
            if selected_model == "other":
                online_provider = Prompt.ask("  Enter provider name (as recognised by LiteLLM visit: https://models.litellm.ai/ for best compatibility)")
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
        auth_requirements = validate_environment(model)

        if not auth_requirements["keys_in_environment"]:
            while not api_key:
                raw_key = Prompt.ask(f"  API key for {provider_name} (will be set as environment variable for this provider)")
                api_key = raw_key.strip()
                if not api_key:
                    console.print("[red]  API key cannot be empty. Please try again.[/red]")
    else:
        # Ask for optional API key for local providers, since some self-hosted servers require it even if they ignore the value.
        raw_key = Prompt.ask(f"  Optional API key for {provider_name} (some local servers require an API key even if they ignore the value)", default="")
        api_key = raw_key.strip() or None

    provider_cfg = ProviderConfig(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # ---- Save ---------------------------------------------------------
    try:
        saved_path = save_model(name, provider_cfg)
    except Exception as exc:
        console.print(f"[red]Failed to save model configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Model '{name}' saved successfully on {saved_path}![/bold green]\n"
    )


@add_app.command("channel")
def add_channel() -> None:
    """Add a new channel (messaging client) configuration."""
    _print_banner()
    console.print("\n[bold green]Add a new channel configuration[/bold green]\n")

    # ---- Channel name ------------------------------------------------
    name = Prompt.ask("[bold]Give this channel a name[/bold]")
    name = name.strip()
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    # Check if channel already exists
    if name in list_channels():
        if not Confirm.ask(f"[yellow]Channel '{name}' already exists. Overwrite?[/yellow]", default=False):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    # ---- Client type --------------------------------------------------
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
    try:
        saved_path = save_channel(name, client_cfg)
    except Exception as exc:
        console.print(f"[red]Failed to save channel configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Channel '{name}' saved successfully![/bold green]\n"
    )


# ---------------------------------------------------------------------------
# DELETE Commands
# ---------------------------------------------------------------------------


@delete_app.command("model")
def delete_model_cmd(
    name: Optional[str] = typer.Argument(None, help="Name of the model to delete"),
) -> None:
    """Delete a saved model configuration."""
    models = list_models()
    if not models:
        console.print("[yellow]No models found. Run [bold]llm-expose add model[/bold] to create one.[/yellow]")
        raise typer.Exit()

    # Select model to delete
    if name is None:
        name = _select_from_list("Select model to delete:", models)
    elif name not in models:
        console.print(f"[red]No model named '{name}' found.[/red]")
        console.print("Run [bold]llm-expose add model[/bold] to see available models.")
        raise typer.Exit(code=1)
    
    # Confirm deletion
    if not Confirm.ask(f"[bold red]Are you sure you want to delete model '{name}'?[/bold red]", default=False):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        raise typer.Exit()

    try:
        delete_model(name)
        console.print(f"\n[bold green]✓ Model '{name}' deleted successfully.[/bold green]\n")
    except FileNotFoundError:
        console.print(f"[red]No model named '{name}' found.[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to delete model '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc


@delete_app.command("channel")
def delete_channel_cmd(
    name: Optional[str] = typer.Argument(None, help="Name of the channel to delete"),
) -> None:
    """Delete a saved channel configuration."""
    channels = list_channels()
    if not channels:
        console.print("[yellow]No channels found. Run [bold]llm-expose add channel[/bold] to create one.[/yellow]")
        raise typer.Exit()

    # Select channel to delete
    if name is None:
        name = _select_from_list("Select channel to delete:", channels)
    elif name not in channels:
        console.print(f"[red]No channel named '{name}' found.[/red]")
        console.print("Run [bold]llm-expose add channel[/bold] to see available channels.")
        raise typer.Exit(code=1)
    
    # Confirm deletion
    if not Confirm.ask(f"[bold red]Are you sure you want to delete channel '{name}'?[/bold red]", default=False):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        raise typer.Exit()

    try:
        delete_channel(name)
        console.print(f"\n[bold green]✓ Channel '{name}' deleted successfully.[/bold green]\n")
    except FileNotFoundError:
        console.print(f"[red]No channel named '{name}' found.[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to delete channel '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# LIST Command
# ---------------------------------------------------------------------------


@list_app.command("models")
@list_app.command("model")
def list_models_cmd() -> None:
    """List all saved model configurations."""
    models = list_models()
    if not models:
        console.print("[yellow]No models found. Run [bold]llm-expose add model[/bold] to create one.[/yellow]")
        return

    table = Table(title="Saved Models", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="blue")
    table.add_column("API Key", style="red")

    for idx, name in enumerate(models, start=1):
        try:
            cfg = load_model(name)
            api_key_offuscated = cfg.api_key[:4] + "****" if cfg.api_key else "-"
            table.add_row(str(idx), name, cfg.provider_name, cfg.model, api_key_offuscated)
        except Exception:
            table.add_row(str(idx), name, "[red]error[/red]", "-", "-")

    console.print(table)


@list_app.command("channels")
@list_app.command("channel")
def list_channels_cmd() -> None:
    """List all saved channel configurations."""
    channels = list_channels()
    if not channels:
        console.print("[yellow]No channels found. Run [bold]llm-expose add channel[/bold] to create one.[/yellow]")
        return

    table = Table(title="Saved Channels", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Type", style="magenta")
    table.add_column("Bot Token", style="red")

    for idx, name in enumerate(channels, start=1):
        try:
            cfg = load_channel(name)
            api_key_offuscated = cfg.bot_token[:4] + "****" if cfg.bot_token else "-"
            table.add_row(str(idx), name, cfg.client_type, api_key_offuscated)
        except Exception:
            table.add_row(str(idx), name, "[red]error[/red]", "-")

    console.print(table)


# ---------------------------------------------------------------------------
# START Command
# ---------------------------------------------------------------------------


@app.command()
def start() -> None:
    """Start the LLM exposure service by selecting a model and channel."""
    _print_banner()
    console.print("\n[bold green]Start LLM Exposure Service[/bold green]\n")

    # ---- Select model -------------------------------------------------
    models = list_models()
    if not models:
        console.print("[red]No models found. Please run [bold]llm-expose add model[/bold] first.[/red]")
        raise typer.Exit(code=1)

    # Show available models in a table
    model_table = Table(title="Available Models", border_style="cyan", show_lines=True)
    model_table.add_column("#", style="dim", width=4)
    model_table.add_column("Name", style="bold")
    model_table.add_column("Provider", style="green")
    model_table.add_column("Model", style="blue")
    model_table.add_column("API Key", style="red")

    for idx, name in enumerate(models, start=1):
        try:
            cfg = load_model(name)
            api_key_offuscated = cfg.api_key[:4] + "****" if cfg.api_key else "-"
            model_table.add_row(
                str(idx),
                name,
                cfg.provider_name,
                cfg.model,
                api_key_offuscated
            )
        except Exception:
            model_table.add_row(str(idx), name, "[red]error[/red]", "-")

    console.print(model_table)
    model_name = _select_from_list("Select a model:", models)

    try:
        provider_cfg = load_model(model_name)
    except Exception as exc:
        console.print(f"[red]Failed to load model '{model_name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # ---- Select channel -----------------------------------------------
    channels = list_channels()
    if not channels:
        console.print("[red]No channels found. Please run [bold]llm-expose add channel[/bold] first.[/red]")
        raise typer.Exit(code=1)

    # Show available channels in a table
    channel_table = Table(title="Available Channels", border_style="cyan", show_lines=True)
    channel_table.add_column("#", style="dim", width=4)
    channel_table.add_column("Name", style="bold")
    channel_table.add_column("Type", style="magenta")
    channel_table.add_column("Bot Token", style="red")

    for idx, name in enumerate(channels, start=1):
        try:
            cfg = load_channel(name)
            bot_token_offuscated = cfg.bot_token[:4] + "****" if cfg.bot_token else "-"
            channel_table.add_row(
                str(idx),
                name,
                cfg.client_type,
                bot_token_offuscated
            )
        except Exception:
            channel_table.add_row(str(idx), name, "[red]error[/red]", "-")

    console.print("\n")
    console.print(channel_table)
    channel_name = _select_from_list("Select a channel:", channels)

    try:
        client_cfg = load_channel(channel_name)
    except Exception as exc:
        console.print(f"[red]Failed to load channel '{channel_name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    system_prompt = None
    #  Ask for load a custom system prompt or use the default one
    if Confirm.ask("\n[bold]Do you want to set a custom system prompt?[/bold]", default=False):
        # Load promt from text file
        while True:
            prompt_path = Prompt.ask("  Enter path to system prompt text file")
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
                break
            except Exception as exc:
                console.print(f"[red]Failed to load system prompt from '{prompt_path}': {exc}[/red]")
                if not Confirm.ask("Do you want to try again?", default=True):
                    system_prompt = None
                    break
        
        console.print(f"\n[green]Custom system prompt loaded successfully![/green]")

    # ---- Display summary and confirm ----------------------------------
    console.print("\n[bold cyan]Selected Configuration:[/bold cyan]")
    summary_table = Table(border_style="cyan", show_header=False)
    summary_table.add_column("Key", style="bold")
    summary_table.add_column("Value")

    summary_table.add_row("Model", model_name)
    summary_table.add_row("Provider", provider_cfg.provider_name)
    summary_table.add_row("Model ID", provider_cfg.model)
    summary_table.add_row("Channel", channel_name)
    summary_table.add_row("Client Type", client_cfg.client_type)
    summary_table.add_row("System Prompt", system_prompt if system_prompt else "[dim]default[/dim]")
    console.print(summary_table)

    if not Confirm.ask("\n[bold]Start the service?[/bold]"):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    # ---- Build and start service --------------------------------------
    # Create a temporary ExposureConfig for the orchestrator
    exposure_name = f"{model_name}_{channel_name}"
    config = ExposureConfig(name=exposure_name, provider=provider_cfg, client=client_cfg, system_prompt=system_prompt)
    
    _start_service(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
