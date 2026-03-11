"""CLI entry point for llm-expose using Typer and Rich."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from telegram.error import Conflict

from llm_expose.config import (
    MCPServerConfig,
    MCPSettingsConfig,
    ProviderConfig,
    TelegramClientConfig,
    add_pair as add_channel_pair,
    delete_channel,
    delete_pair as delete_channel_pair,
    delete_mcp_server,
    delete_model,
    get_pairs_for_channel,
    get_mcp_server,
    list_channels,
    list_mcp_servers,
    list_models,
    list_pairs,
    load_channel,
    load_mcp_config,
    load_mcp_settings,
    load_model,
    save_channel,
    save_mcp_server,
    save_mcp_settings,
    save_model,
)
from llm_expose.config.models import ExposureConfig
from llm_expose.core.builtin_mcp import ToolExecutionContext
from llm_expose.core.content_parts import (
    build_local_attachment_descriptor,
    build_user_content,
    file_to_data_url,
)
from llm_expose.core.orchestrator import Orchestrator
from llm_expose.core.tool_aware_completion import ToolAwareCompletion
from llm_expose.providers.litellm_provider import LiteLLMProvider
from llm_expose.clients.telegram import TelegramClient
from llm_expose.clients.base import BaseClient

logger = logging.getLogger(__name__)

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
add_app = typer.Typer(help="Add model, channel, pairs, or MCP server configurations")
delete_app = typer.Typer(help="Delete model, channel, pairs, or MCP server configurations")
list_app = typer.Typer(help="List saved models, channels, pairs, or MCP servers")

app.add_typer(add_app, name="add")
app.add_typer(delete_app, name="delete")
app.add_typer(list_app, name="list")

console = Console()

# ---------------------------------------------------------------------------
# ASCII art / branding
# ---------------------------------------------------------------------------

_BANNER = r"""
   |\---/|
   | ,_, |
    \_`_/-..----.
 ___/ `   ' ,""+ \ 
(__...'   __\    |`.___.';
  (_,...'(_,.`__)/'.....+                        
╦  ╦  ╔╦╗  ╔═╗─┐ ┬┌─┐┌─┐┌─┐┌─┐
║  ║  ║║║  ║╣ ┌┴┬┘├─┘│ │└─┐├┤ 
╩═╝╩═╝╩ ╩  ╚═╝┴ └─┴  └─┘└─┘└─┘

> LLM's at your service, wherever you chat.
> v0.1-alpha
> 100+ models supported via LiteLLM and local OpenAI-compatible servers.
> MCP support for advanced tool integration and control.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Display the welcome banner."""
    console.print(Panel(Text(_BANNER, style="bold cyan"), border_style="cyan"))


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


def _parse_multi_select_numbers(raw: str) -> list[int] | None:
    """Parse comma-separated positive integers preserving order and uniqueness."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None

    values: list[int] = []
    seen: set[int] = set()
    for part in parts:
        if not part.isdigit():
            return None
        value = int(part)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _select_mcp_servers_for_channel(options: list[str]) -> list[str]:
    """Select one or many MCP servers to attach to a channel."""
    if not options:
        console.print("\n[yellow]No MCP servers configured. This channel will start with none attached.[/yellow]")
        return []

    all_index = len(options) + 1
    none_index = len(options) + 2

    console.print("\n[bold]Available MCP servers:[/bold]")
    for idx, option in enumerate(options, start=1):
        console.print(f"  [cyan]{idx})[/cyan] {option}")
    console.print(f"  [cyan]{all_index})[/cyan] Select all")
    console.print(f"  [cyan]{none_index})[/cyan] Select none")

    while True:
        raw = Prompt.ask("  Enter number(s), comma-separated")
        selected_indexes = _parse_multi_select_numbers(raw)
        if selected_indexes is None:
            console.print("[red]  Invalid input. Use numbers separated by commas (example: 1,3).[/red]")
            continue

        if all_index in selected_indexes and len(selected_indexes) == 1:
            return options.copy()
        if none_index in selected_indexes and len(selected_indexes) == 1:
            return []
        if all_index in selected_indexes or none_index in selected_indexes:
            console.print("[red]  'Select all' and 'Select none' must be used alone.[/red]")
            continue

        if any(index < 1 or index > len(options) for index in selected_indexes):
            console.print("[red]  Invalid selection. Please pick numbers from the list.[/red]")
            continue

        return [options[index - 1] for index in selected_indexes]


def _resolve_channel_name_for_pairs(channel_name: Optional[str]) -> str:
    """Resolve the channel config name used for pair CRUD commands."""
    channels = list_channels()
    if not channels:
        console.print(
            "[yellow]No channels found. Run [bold]llm-expose add channel[/bold] first.[/yellow]"
        )
        raise typer.Exit(code=1)

    if channel_name is None:
        return _select_from_list("Select a channel:", channels)

    normalized = channel_name.strip()
    if normalized not in channels:
        console.print(f"[red]No channel named '{normalized}' found.[/red]")
        raise typer.Exit(code=1)
    return normalized


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
        from litellm import validate_environment, model_cost

        models_by_provider = {}

        # model cost schema from https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
        for key, value in model_cost.items():
            provider = value.get("litellm_provider", "unknown").lower()
            mode = value.get("mode", "unknown").lower()

            if not mode in ("chat", "completion"):
                continue
            
            model_supports = []
            model_supports += ["[green]tools[/green]"] if value.get("supports_function_calling", False) else []
            model_supports += ["[yellow]vision[/yellow]"] if value.get("supports_vision", False) else []
            model_supports = f" ({', '.join(model_supports)})" if model_supports else ""

            if provider not in models_by_provider:
                models_by_provider[provider] = []

            models_by_provider[provider].append(key + model_supports)

        online_provider = _select_from_list(
            "Select online provider:",
            list(models_by_provider.keys()),
        )
        # Show available models for this provider
        available_models = list(models_by_provider[online_provider])
        selected_model = _select_from_list(
            f"Select model for {online_provider}:",
            available_models,
        )

        # Remove model supports info
        selected_model = selected_model.split()[0].strip()
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

    # ---- Model selection for --llm-completion ---------------------------
    model_name: Optional[str] = None
    available_models = list_models()
    if available_models or Confirm.ask("\n[bold]Do you want to set a model for LLM completion?[/bold]", default=False):
        if available_models:
            model_options = available_models + ["None (skip LLM completion)"]
            selected_model = _select_from_list(
                "Select a model for this channel (used with --llm-completion):",
                model_options,
            )
            if selected_model != "None (skip LLM completion)":
                model_name = selected_model
        else:
            console.print("[yellow]No models found. Run 'llm-expose add model' first to create one.[/yellow]")

    available_mcp_servers = list_mcp_servers()
    attached_mcp_servers = _select_mcp_servers_for_channel(available_mcp_servers)

    system_prompt_path: Optional[str] = None
    if Confirm.ask("\n[bold]Do you want to set a custom system prompt for this channel?[/bold]", default=False):
        while True:
            prompt_path = Prompt.ask("  Enter path to system prompt text file")
            try:
                # Verify the file exists and is readable
                with open(prompt_path, "r", encoding="utf-8") as f:
                    f.read()  # Just verify we can read it
                system_prompt_path = prompt_path
                break
            except Exception as exc:
                console.print(f"[red]Failed to access system prompt file '{prompt_path}': {exc}[/red]")
                if not Confirm.ask("Do you want to try again?", default=True):
                    system_prompt_path = None
                    break

        if system_prompt_path:
            console.print("\n[green]System prompt file path configured successfully![/green]")

    client_cfg = TelegramClientConfig(
        bot_token=bot_token,
        mcp_servers=attached_mcp_servers,
        system_prompt_path=system_prompt_path,
        model_name=model_name,
    )

    # ---- Save ---------------------------------------------------------
    try:
        saved_path = save_channel(name, client_cfg)
    except Exception as exc:
        console.print(f"[red]Failed to save channel configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Channel '{name}' saved successfully![/bold green]\n"
    )


@add_app.command("pair")
def add_pair_cmd(
    pair_id: Optional[str] = typer.Argument(
        None,
        help="Pair ID (for Telegram this is the chat ID)",
    ),
    channel: Optional[str] = typer.Option(
        None,
        "--channel",
        "-c",
        help="Channel config name to attach this pair to",
    ),
) -> None:
    """Add a pair ID to a channel allowlist."""
    channel_name = _resolve_channel_name_for_pairs(channel)

    resolved_pair_id = pair_id.strip() if pair_id is not None else ""
    while not resolved_pair_id:
        resolved_pair_id = Prompt.ask("Pair ID (for Telegram this is chat.id)").strip()
        if not resolved_pair_id:
            console.print("[red]Pair ID cannot be empty.[/red]")

    try:
        saved_path = add_channel_pair(channel_name, resolved_pair_id)
    except Exception as exc:
        console.print(f"[red]Failed to add pair: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Pair '{resolved_pair_id}' added to channel '{channel_name}' ({saved_path}).[/bold green]\n"
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


@delete_app.command("pair")
def delete_pair_cmd(
    pair_id: Optional[str] = typer.Argument(
        None,
        help="Pair ID to delete",
    ),
    channel: Optional[str] = typer.Option(
        None,
        "--channel",
        "-c",
        help="Channel config name that owns this pair",
    ),
) -> None:
    """Delete a pair ID from a channel allowlist."""
    channel_name = _resolve_channel_name_for_pairs(channel)

    resolved_pair_id = pair_id.strip() if pair_id is not None else ""
    if not resolved_pair_id:
        available_pairs = get_pairs_for_channel(channel_name)
        if not available_pairs:
            console.print(
                f"[yellow]No pairs found for channel '{channel_name}'.[/yellow]"
            )
            raise typer.Exit(code=1)
        resolved_pair_id = _select_from_list("Select pair to delete:", available_pairs)

    try:
        delete_channel_pair(channel_name, resolved_pair_id)
    except FileNotFoundError:
        console.print(
            f"[red]No pair '{resolved_pair_id}' found for channel '{channel_name}'.[/red]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to delete pair: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ Pair '{resolved_pair_id}' removed from channel '{channel_name}'.[/bold green]\n"
    )


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
    table.add_column("MCP Attached", style="cyan")

    for idx, name in enumerate(channels, start=1):
        try:
            cfg = load_channel(name)
            api_key_offuscated = cfg.bot_token[:4] + "****" if cfg.bot_token else "-"
            attached = ", ".join(cfg.mcp_servers) if cfg.mcp_servers else "none"
            table.add_row(str(idx), name, cfg.client_type, api_key_offuscated, attached)
        except Exception:
            table.add_row(str(idx), name, "[red]error[/red]", "-", "-")

    console.print(table)


@list_app.command("pairs")
@list_app.command("pair")
def list_pairs_cmd(
    channel: Optional[str] = typer.Option(
        None,
        "--channel",
        "-c",
        help="Filter by channel config name",
    ),
) -> None:
    """List channel pairing allowlists."""
    pair_map = list_pairs(channel)
    visible = [(name, pair_ids) for name, pair_ids in pair_map.items() if pair_ids]
    if not visible:
        console.print(
            "[yellow]No pairs found. Run [bold]llm-expose add pair <id>[/bold] to add one.[/yellow]"
        )
        return

    table = Table(title="Configured Pairs", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Channel", style="bold")
    table.add_column("Pair ID", style="green")

    row = 1
    for channel_name, pair_ids in sorted(visible):
        for pair_value in pair_ids:
            table.add_row(str(row), channel_name, pair_value)
            row += 1

    console.print(table)


# ---------------------------------------------------------------------------
# MCP Commands
# ---------------------------------------------------------------------------


@list_app.command("mcp")
@list_app.command("mcps")
def list_mcp_cmd() -> None:
    """List all configured MCP servers and global MCP settings."""
    settings = load_mcp_settings()
    servers = list_mcp_servers()

    console.print("\n[bold cyan]MCP Settings[/bold cyan]")
    settings_table = Table(border_style="cyan", show_header=False)
    settings_table.add_column("Key", style="bold")
    settings_table.add_column("Value")
    settings_table.add_row("confirmation_mode", settings.confirmation_mode)
    settings_table.add_row("tool_timeout_seconds", str(settings.tool_timeout_seconds))
    console.print(settings_table)

    if not servers:
        console.print(
            "\n[yellow]No MCP servers configured. Use [bold]llm-expose add mcp[/bold] to add one.[/yellow]"
        )
        return

    table = Table(title="Configured MCP Servers", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Transport", style="green")
    table.add_column("Target", style="blue")
    table.add_column("Args", style="magenta")
    table.add_column("Confirmation", style="yellow")

    for idx, name in enumerate(servers, start=1):
        try:
            server = get_mcp_server(name)
            if server.transport == "stdio":
                target = server.command
            elif server.transport in {"sse", "http"}:
                target = server.url
            else:
                target = "[builtin]"
            table.add_row(
                str(idx),
                server.name,
                server.transport,
                target or "-",
                " ".join(list(server.args)),
                server.tool_confirmation,
            )
        except Exception:
            table.add_row(str(idx), name, "[red]error[/red]", "-", "-", "-")

    console.print("\n")
    console.print(table)


@add_app.command("mcp")
def add_mcp_cmd() -> None:
    """Add or update an MCP server configuration."""
    _print_banner()
    console.print("\n[bold green]Add an MCP server configuration[/bold green]\n")

    name = Prompt.ask("[bold]MCP server name[/bold]").strip()
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    existing_servers = list_mcp_servers()
    if name in existing_servers:
        if not Confirm.ask(
            f"[yellow]MCP server '{name}' already exists. Overwrite?[/yellow]",
            default=False,
        ):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    transport = _select_from_list("Select MCP transport:", ["stdio", "sse"])
    command: Optional[str] = None
    args: list[str] = []
    url: Optional[str] = None

    if transport == "stdio":
        command = Prompt.ask("  Command to run (example: npx, uvx)").strip()
        if not command:
            console.print("[red]Command cannot be empty for stdio transport.[/red]")
            raise typer.Exit(code=1)
        raw_args = Prompt.ask("  Command args (space-separated)", default="").strip()
        args = raw_args.split() if raw_args else []
    else:
        url = Prompt.ask("  SSE URL (example: http://localhost:3000/sse)").strip()
        if not url:
            console.print("[red]URL cannot be empty for SSE transport.[/red]")
            raise typer.Exit(code=1)

    enabled = Confirm.ask("  Enable this MCP server now?", default=True)

    # Tool confirmation mode
    console.print("\n[bold cyan]Tool Confirmation Mode:[/bold cyan]")
    console.print("  [dim]• default: Use global confirmation setting[/dim]")
    console.print("  [dim]• required: Always require user approval before executing tools[/dim]")
    console.print("  [dim]• never: Always auto-execute tools without approval[/dim]\n")
    
    tool_confirmation = _select_from_list(
        "Select tool confirmation mode:",
        ["default", "required", "never"]
    )

    server = MCPServerConfig(
        name=name,
        transport=transport,
        command=command,
        args=args,
        url=url,
        enabled=enabled,
        tool_confirmation=tool_confirmation,
    )

    try:
        path = save_mcp_server(server)
    except Exception as exc:
        console.print(f"[red]Failed to save MCP server configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"\n[bold green]✓ MCP server '{name}' saved successfully in {path}![/bold green]\n"
    )


@delete_app.command("mcp")
def delete_mcp_cmd(
    name: Optional[str] = typer.Argument(None, help="Name of the MCP server to delete"),
) -> None:
    """Delete a configured MCP server."""
    servers = list_mcp_servers()
    if not servers:
        console.print(
            "[yellow]No MCP servers found. Run [bold]llm-expose mcp add[/bold] to create one.[/yellow]"
        )
        raise typer.Exit()

    if name is None:
        name = _select_from_list("Select MCP server to delete:", servers)
    elif name not in servers:
        console.print(f"[red]No MCP server named '{name}' found.[/red]")
        raise typer.Exit(code=1)

    if not Confirm.ask(
        f"[bold red]Are you sure you want to delete MCP server '{name}'?[/bold red]",
        default=False,
    ):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        raise typer.Exit()

    try:
        delete_mcp_server(name)
    except FileNotFoundError:
        console.print(f"[red]No MCP server named '{name}' found.[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Failed to delete MCP server '{name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[bold green]✓ MCP server '{name}' deleted successfully.[/bold green]\n")


# ---------------------------------------------------------------------------
# START Command
# ---------------------------------------------------------------------------


@app.command()
def start() -> None:
    """Start the LLM exposure service by selecting a channel."""
    _print_banner()
    console.print("\n[bold green]Start LLM Exposure Service[/bold green]\n")

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
    channel_table.add_column("Model", style="green")
    channel_table.add_column("Bot Token", style="red")

    for idx, name in enumerate(channels, start=1):
        try:
            cfg = load_channel(name)
            bot_token_offuscated = cfg.bot_token[:4] + "****" if cfg.bot_token else "-"
            model_display = cfg.model_name if cfg.model_name else "[dim]none[/dim]"
            channel_table.add_row(
                str(idx),
                name,
                cfg.client_type,
                model_display,
                bot_token_offuscated
            )
        except Exception:
            channel_table.add_row(str(idx), name, "[red]error[/red]", "-", "-")

    console.print(channel_table)
    channel_name = _select_from_list("Select a channel:", channels)

    try:
        client_cfg = load_channel(channel_name)
    except Exception as exc:
        console.print(f"[red]Failed to load channel '{channel_name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # ---- Load model from channel -----------------------------------------------
    if not client_cfg.model_name:
        console.print(f"[red]Error: Channel '{channel_name}' has no model configured.[/red]")
        console.print(f"[yellow]Tip: Run [bold]llm-expose add channel {channel_name}[/bold] to set a model.[/yellow]")
        raise typer.Exit(code=1)

    model_name = client_cfg.model_name
    try:
        provider_cfg = load_model(model_name)
    except FileNotFoundError:
        console.print(f"[red]Error: Model '{model_name}' not found.[/red]")
        console.print(f"[yellow]Tip: Run [bold]llm-expose list model[/bold] to see available models.[/yellow]")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"[red]Failed to load model '{model_name}': {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # ---- Display summary and confirm ----------------------------------
    console.print("\n[bold cyan]Selected Configuration:[/bold cyan]")
    summary_table = Table(border_style="cyan", show_header=False)
    summary_table.add_column("Key", style="bold")
    summary_table.add_column("Value")

    summary_table.add_row("Channel", channel_name)
    summary_table.add_row("Client Type", client_cfg.client_type)
    summary_table.add_row("Model", model_name)
    summary_table.add_row("Provider", provider_cfg.provider_name)
    summary_table.add_row("Model ID", provider_cfg.model)
    summary_table.add_row(
        "MCP Attached",
        ", ".join(client_cfg.mcp_servers) if client_cfg.mcp_servers else "[dim]none[/dim]",
    )
    summary_table.add_row(
        "System Prompt Path",
        client_cfg.system_prompt_path if client_cfg.system_prompt_path else "[dim]none (using default)[/dim]",
    )
    console.print(summary_table)

    if client_cfg.mcp_servers:
        console.print(
            f"\n[bold cyan]MCP attached to this channel ({len(client_cfg.mcp_servers)}):[/bold cyan] "
            + ", ".join(client_cfg.mcp_servers)
        )
    else:
        console.print("\n[bold cyan]MCP attached to this channel:[/bold cyan] none")

    # Check if system prompt file exists (if configured)
    if client_cfg.system_prompt_path:
        from pathlib import Path
        prompt_file = Path(client_cfg.system_prompt_path)
        if not prompt_file.exists():
            console.print(
                f"\n[yellow]Warning: System prompt file not found: {client_cfg.system_prompt_path}[/yellow]\n"
                "[yellow]The default system prompt will be used instead.[/yellow]"
            )

    if not Confirm.ask("\n[bold]Start the service?[/bold]"):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    # ---- Build and start service --------------------------------------
    # Create a temporary ExposureConfig for the orchestrator
    exposure_name = f"{model_name}_{channel_name}"
    config = ExposureConfig(
        name=exposure_name,
        channel_name=channel_name,
        provider=provider_cfg,
        client=client_cfg,
    )
    
    _start_service(config)


# ---------------------------------------------------------------------------
# MESSAGE Command
# ---------------------------------------------------------------------------


@app.command()
def message(
    channel: str = typer.Argument(
        ...,
        help="Channel config name to send the message through",
    ),
    user_id: str = typer.Argument(
        ...,
        help="User/Chat ID to send the message to (for Telegram this is the chat.id)",
    ),
    text: str = typer.Argument(
        ...,
        help="Message text to send, or instructions for LLM if --llm-completion is used",
    ),
    llm_completion: bool = typer.Option(
        False,
        "--llm-completion",
        help="Process text as instructions to LLM (generates response to send)",
    ),
    suppress_send: bool = typer.Option(
        False,
        "--suppress-send",
        help="Generate the model response but do not automatically send it to the user (requires --llm-completion)",
    ),
    system_prompt_file: Optional[str] = typer.Option(
        None,
        "--system-prompt-file",
        help="Path to custom system prompt file (overrides channel's system prompt)",
    ),
    image: list[str] = typer.Option(
        [],
        "--image",
        "-i",
        help="Path to an image file to include in LLM input (repeatable).",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a local file to send directly (without LLM processing).",
    ),
) -> None:
    """Send a direct message to a specific user in a channel.
    
    Without --llm-completion, the text is sent directly to the user.
    
    With --llm-completion, the text is treated as instructions to the LLM.
    The LLM uses the channel's system prompt + your instructions to generate
    a response, which is then sent to the user (your instructions are not sent).

    With --llm-completion --suppress-send, the LLM response is generated and
    returned in the JSON output, but no message or image is delivered to the user.

    With --file, the text is sent directly and then the file is sent as a
    document attachment to the same user.
    
    Examples:
        llm-expose message support 12345 "System is down"
        llm-expose message alerts 67890 "CPU at 95%. Generate a brief alert." --llm-completion
        llm-expose message news 12345 "Summarize today's headlines" --llm-completion --system-prompt-file /path/to/prompt.txt
        llm-expose message ops 12345 "Draft a reply and decide later whether to send it" --llm-completion --suppress-send
        llm-expose message ops 12345 "Some file is here:" -f /path/to/file.pdf
    
    This command is useful for cron jobs and scheduled notifications.
    User must be paired with the channel (run 'llm-expose add pair' if needed).
    """
    # When invoked as a plain function (tests/internal calls), Typer option
    # defaults may still be OptionInfo objects instead of concrete values.
    if isinstance(system_prompt_file, typer.models.OptionInfo):
        system_prompt_file = None
    if isinstance(file, typer.models.OptionInfo):
        file = None
    if isinstance(image, typer.models.OptionInfo):
        image = []

    # Validate inputs
    channel = channel.strip()
    user_id = user_id.strip()
    text = text.strip()

    if not channel:
        console.print("[red]Error: Channel name cannot be empty.[/red]")
        raise typer.Exit(code=1)

    if not user_id:
        console.print("[red]Error: User ID cannot be empty.[/red]")
        raise typer.Exit(code=1)

    if not text:
        console.print("[red]Error: Message text cannot be empty.[/red]")
        raise typer.Exit(code=1)

    if suppress_send and not llm_completion:
        console.print("[red]Error: --suppress-send requires --llm-completion.[/red]")
        raise typer.Exit(code=1)

    if file and llm_completion:
        console.print("[red]Error: --file cannot be used with --llm-completion.[/red]")
        raise typer.Exit(code=1)

    file_path: Optional[Path] = None
    if file:
        file_path = Path(file).expanduser()
        if not file_path.exists() or not file_path.is_file():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(code=1)

    # Load channel configuration
    try:
        client_cfg = load_channel(channel)
    except FileNotFoundError:
        console.print(f"[red]Error: Channel '{channel}' not found.[/red]")
        console.print("[yellow]Tip: Run 'llm-expose list channel' to see available channels.[/yellow]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error: Failed to load channel '{channel}': {exc}[/red]")
        raise typer.Exit(code=1)

    # Validate pairing: ensure user_id is paired with this channel
    try:
        paired_users = get_pairs_for_channel(channel)
    except Exception as exc:
        console.print(f"[red]Error: Failed to load pairing configuration: {exc}[/red]")
        raise typer.Exit(code=1)

    if user_id not in paired_users:
        console.print(
            f"[red]Error: User '{user_id}' is not paired with channel '{channel}'.[/red]"
        )
        if paired_users:
            console.print(f"[yellow]Paired users: {', '.join(paired_users)}[/yellow]")
        console.print(
            f"[yellow]Tip: Add this user with: llm-expose add pair {user_id} --channel {channel}[/yellow]"
        )
        raise typer.Exit(code=1)

    # Determine the final message to send
    final_message = text
    llm_response_text = None
    llm_model_used = None
    image_data_urls: list[str] = []
    image_paths: list[Path] = []

    if llm_completion and image:
        for image_path_str in image:
            image_path = Path(image_path_str).expanduser()
            if not image_path.exists() or not image_path.is_file():
                console.print(f"[red]Error: Image file not found: {image_path_str}[/red]")
                raise typer.Exit(code=1)
            image_paths.append(image_path)
            image_data_urls.append(file_to_data_url(image_path))

    # If LLM completion is requested, generate response from instructions
    if llm_completion:
        # Validate channel has a model configured
        if not client_cfg.model_name:
            console.print(f"[red]Error: Channel '{channel}' has no model configured.[/red]")
            console.print(f"[yellow]Tip: Run [bold]llm-expose add channel {channel}[/bold] to set a model.[/yellow]")
            raise typer.Exit(code=1)

        # Load model configuration
        try:
            provider_cfg = load_model(client_cfg.model_name)
        except FileNotFoundError:
            console.print(f"[red]Error: Model '{client_cfg.model_name}' not found.[/red]")
            console.print("[yellow]Tip: Run 'llm-expose list model' to see available models.[/yellow]")
            raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[red]Error: Failed to load model '{client_cfg.model_name}': {exc}[/red]")
            raise typer.Exit(code=1)

        # Determine system prompt (override or channel's)
        system_prompt: Optional[str] = None
        
        # First try the override file if provided
        if system_prompt_file:
            try:
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
                if not system_prompt.strip():
                    console.print(f"[red]Error: System prompt file is empty.[/red]")
                    raise typer.Exit(code=1)
            except FileNotFoundError:
                console.print(f"[red]Error: System prompt file '{system_prompt_file}' not found.[/red]")
                raise typer.Exit(code=1)
            except Exception as exc:
                console.print(f"[red]Error: Failed to read system prompt file: {exc}[/red]")
                raise typer.Exit(code=1)
        # Otherwise try the channel's configured prompt path
        elif client_cfg.system_prompt_path:
            try:
                with open(client_cfg.system_prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
                if not system_prompt.strip():
                    console.print(f"[red]Error: Channel's system prompt file is empty.[/red]")
                    raise typer.Exit(code=1)
            except FileNotFoundError:
                console.print(f"[red]Error: Channel's system prompt file '{client_cfg.system_prompt_path}' not found.[/red]")
                raise typer.Exit(code=1)
            except Exception as exc:
                console.print(f"[red]Error: Failed to read channel's system prompt file: {exc}[/red]")
                raise typer.Exit(code=1)

        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Respond concisely and professionally."

        # Call LLM to generate response
        try:
            provider = LiteLLMProvider(provider_cfg)
            
            # Check if MCP servers are configured for tool-aware completion
            if client_cfg.mcp_servers:
                # Tool-aware completion with MCP
                try:
                    mcp_config = load_mcp_config()
                    user_content = build_user_content(text, image_urls=image_data_urls)
                    expose_paths = mcp_config.settings.expose_attachment_paths
                    attachment_paths_by_ref: dict[str, str] = {}
                    invocation_attachments: list[dict[str, Any]] = []
                    for image_path in image_paths:
                        attachment_ref = f"att_{uuid.uuid4().hex}"
                        attachment_paths_by_ref[attachment_ref] = str(image_path.resolve())
                        invocation_attachments.append(
                            build_local_attachment_descriptor(
                                image_path,
                                kind="image",
                                include_path=expose_paths,
                                attachment_ref=attachment_ref,
                            )
                        )
                    # --suppress-send only disables the CLI's final auto-delivery.
                    # Tool calls still require a sender to execute explicit send actions.
                    tool_sender = TelegramClient(client_cfg, handler=_placeholder_handler)
                    execution_context = ToolExecutionContext(
                        execution_mode="one-shot",
                        channel_id=user_id,
                        channel_name=channel,
                        subject_id=user_id,
                        subject_kind="chat",
                        platform=client_cfg.client_type,
                        attachments=invocation_attachments,
                        attachment_paths_by_ref=attachment_paths_by_ref,
                        sender=tool_sender,
                    )
                    async def _tool_aware_complete():
                        async with ToolAwareCompletion(
                            provider=provider,
                            mcp_config=mcp_config,
                            requested_servers=client_cfg.mcp_servers,
                            timeout_seconds=30,
                        ) as handler:
                            return await handler.complete(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_content},
                                ],
                                execution_context=execution_context,
                            )
                    llm_response_text = asyncio.run(_tool_aware_complete())
                except Exception as exc:
                    console.print(f"[red]Error: Tool-aware completion failed: {exc}[/red]")
                    logger.exception("Tool-aware completion error")
                    raise typer.Exit(code=1)
            else:
                # Simple completion without tools
                user_content = build_user_content(text, image_urls=image_data_urls)
                llm_response_text = asyncio.run(
                    provider.complete(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ]
                    )
                )
            
            llm_model_used = client_cfg.model_name
            final_message = llm_response_text
            logger.info(
                "LLM completion: generated response for user %s in channel %s using model %s",
                user_id,
                channel,
                client_cfg.model_name,
            )
        except Exception as exc:
            console.print(f"[red]Error: LLM failed to generate response: {exc}[/red]")
            logger.exception("LLM completion error")
            raise typer.Exit(code=1)
    elif system_prompt_file:
        # Warn if system_prompt_file is provided without --llm-completion
        console.print(
            "[yellow]Warning: --system-prompt-file is ignored without --llm-completion.[/yellow]"
        )
    elif image:
        console.print(
            "[yellow]Warning: --image is ignored without --llm-completion.[/yellow]"
        )

    # Create client instance and send message unless auto-delivery is suppressed
    try:
        if suppress_send:
            result = {
                "status": "suppressed",
                "channel": channel,
                "user_id": user_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        else:
            # Client requires a handler, but send_message() doesn't use it
            client = TelegramClient(client_cfg, handler=_placeholder_handler)

            # Keep a single event loop for all send operations in this CLI invocation.
            async def _send_all() -> dict:
                send_result = await client.send_message(user_id, final_message)

                if file_path is not None:
                    file_result = await client.send_file(user_id, str(file_path))
                    send_result["file_reference"] = file_result

                if llm_completion and image_data_urls:
                    image_result = await client.send_images(user_id, image_data_urls[:1])
                    send_result["image_reference"] = image_result

                return send_result

            send_coro = _send_all()
            try:
                result = asyncio.run(send_coro)
            finally:
                # In tests, asyncio.run may be mocked and not await the coroutine.
                if inspect.iscoroutine(send_coro) and send_coro.cr_frame is not None:
                    send_coro.close()
        
        # Extend result with LLM metadata if applicable
        if llm_completion and llm_response_text:
            result["llm_response"] = llm_response_text
            result["llm_model"] = llm_model_used
        
        # Output result as JSON for cron integration
        console.print(json.dumps(result, indent=2))
        
    except Exception as exc:
        console.print(f"[red]Error: Failed to send message: {exc}[/red]")
        logger.exception("Message send error")
        raise typer.Exit(code=1)


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


async def _placeholder_handler(channel_id: str, message: str) -> str:  # pragma: no cover
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
