"""Configuration loading and saving utilities for llm-expose."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from llm_expose.config.models import (
    MCPConfig,
    MCPServerConfig,
    MCPSettingsConfig,
    PairingsConfig,
    ProviderConfig,
    TelegramClientConfig,
)

# Default base directory for all configs
_DEFAULT_BASE_DIR = Path.home() / ".llm-expose"


def get_base_dir() -> Path:
    """Return the base directory used to store all configs.

    The directory is resolved from the ``LLM_EXPOSE_CONFIG_DIR`` environment
    variable when set, otherwise defaults to ``~/.llm-expose``.
    """
    env_dir = os.environ.get("LLM_EXPOSE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_BASE_DIR


def get_models_dir() -> Path:
    """Return the directory used to store model configs."""
    return get_base_dir() / "models"


def get_channels_dir() -> Path:
    """Return the directory used to store channel configs."""
    return get_base_dir() / "channels"


def get_mcp_config_path() -> Path:
    """Return the file path used to store MCP server settings/config."""
    return get_base_dir() / "mcp_servers.yaml"


def get_pairs_config_path() -> Path:
    """Return the file path used to store channel pairing settings."""
    return get_base_dir() / "pairs.yaml"


# ---------------------------------------------------------------------------
# Model (Provider) Config Management
# ---------------------------------------------------------------------------


def save_model(name: str, config: ProviderConfig) -> Path:
    """Persist a model configuration to a YAML file.

    Args:
        name: Unique name for this model configuration.
        config: The :class:`ProviderConfig` instance to save.

    Returns:
        The :class:`~pathlib.Path` where the config was written.
    """
    directory = get_models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    data = {"name": name, **config.model_dump()}
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return path


def load_model(name: str) -> ProviderConfig:
    """Load a model configuration from disk by name.

    Args:
        name: The name used when the model was saved.

    Returns:
        The deserialized :class:`ProviderConfig`.

    Raises:
        FileNotFoundError: If no model with the given name exists.
        ValueError: If the YAML is malformed or validation fails.
    """
    path = get_models_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model configuration named '{name}' found")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # Remove the 'name' key as it's not part of ProviderConfig
    data.pop("name", None)
    return ProviderConfig.model_validate(data)


def list_models() -> list[str]:
    """Return names of all saved model configs.

    Returns:
        A sorted list of model config names.
    """
    directory = get_models_dir()
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def delete_model(name: str) -> None:
    """Delete a saved model config by name.

    Args:
        name: The name of the model config to delete.

    Raises:
        FileNotFoundError: If no model with the given name exists.
    """
    path = get_models_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model configuration named '{name}' found")
    path.unlink()


# ---------------------------------------------------------------------------
# Channel (Client) Config Management
# ---------------------------------------------------------------------------


def save_channel(name: str, config: TelegramClientConfig) -> Path:
    """Persist a channel configuration to a YAML file.

    Args:
        name: Unique name for this channel configuration.
        config: The :class:`TelegramClientConfig` instance to save.

    Returns:
        The :class:`~pathlib.Path` where the config was written.
    """
    directory = get_channels_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    data = {"name": name, **config.model_dump()}
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return path


def load_channel(name: str) -> TelegramClientConfig:
    """Load a channel configuration from disk by name.

    Args:
        name: The name used when the channel was saved.

    Returns:
        The deserialized :class:`TelegramClientConfig`.

    Raises:
        FileNotFoundError: If no channel with the given name exists.
        ValueError: If the YAML is malformed or validation fails.
    """
    path = get_channels_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No channel configuration named '{name}' found")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # Remove the 'name' key as it's not part of TelegramClientConfig
    data.pop("name", None)
    return TelegramClientConfig.model_validate(data)


def list_channels() -> list[str]:
    """Return names of all saved channel configs.

    Returns:
        A sorted list of channel config names.
    """
    directory = get_channels_dir()
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def delete_channel(name: str) -> None:
    """Delete a saved channel config by name.

    Args:
        name: The name of the channel config to delete.

    Raises:
        FileNotFoundError: If no channel with the given name exists.
    """
    path = get_channels_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No channel configuration named '{name}' found")
    path.unlink()


# ---------------------------------------------------------------------------
# MCP Server Config Management
# ---------------------------------------------------------------------------


def load_mcp_config() -> MCPConfig:
    """Load MCP configuration from disk.

    Returns defaults when the config file does not exist yet.
    """
    path = get_mcp_config_path()
    if not path.exists():
        return MCPConfig()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return MCPConfig.model_validate(data)


def save_mcp_config(config: MCPConfig) -> Path:
    """Persist MCP configuration to disk.

    Args:
        config: Full MCP configuration object.

    Returns:
        The path where the YAML file was written.
    """
    path = get_mcp_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, allow_unicode=True, sort_keys=False)
    return path


def list_mcp_servers() -> list[str]:
    """Return names of configured MCP servers."""
    config = load_mcp_config()
    return sorted(server.name for server in config.servers)


def get_mcp_server(name: str) -> MCPServerConfig:
    """Return a single MCP server config by name.

    Raises:
        FileNotFoundError: If no MCP server with the given name exists.
    """
    config = load_mcp_config()
    for server in config.servers:
        if server.name == name:
            return server
    raise FileNotFoundError(f"No MCP server named '{name}' found")


def save_mcp_server(server: MCPServerConfig) -> Path:
    """Create or update a single MCP server entry by name."""
    config = load_mcp_config()
    updated = False
    for idx, current in enumerate(config.servers):
        if current.name == server.name:
            config.servers[idx] = server
            updated = True
            break
    if not updated:
        config.servers.append(server)
    return save_mcp_config(config)


def delete_mcp_server(name: str) -> None:
    """Delete a configured MCP server by name.

    Raises:
        FileNotFoundError: If no MCP server with the given name exists.
    """
    config = load_mcp_config()
    original_count = len(config.servers)
    config.servers = [server for server in config.servers if server.name != name]
    if len(config.servers) == original_count:
        raise FileNotFoundError(f"No MCP server named '{name}' found")
    save_mcp_config(config)


def load_mcp_settings() -> MCPSettingsConfig:
    """Return global MCP runtime settings."""
    return load_mcp_config().settings


def save_mcp_settings(settings: MCPSettingsConfig) -> Path:
    """Persist only global MCP runtime settings."""
    config = load_mcp_config()
    config.settings = settings
    return save_mcp_config(config)


# ---------------------------------------------------------------------------
# Channel Pairing Config Management
# ---------------------------------------------------------------------------


def load_pairings_config() -> PairingsConfig:
    """Load channel pairing configuration from disk.

    Returns defaults when the config file does not exist yet.
    """
    path = get_pairs_config_path()
    if not path.exists():
        return PairingsConfig()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return PairingsConfig.model_validate(data)


def save_pairings_config(config: PairingsConfig) -> Path:
    """Persist channel pairing configuration to disk."""
    path = get_pairs_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, allow_unicode=True, sort_keys=False)
    return path


def list_pairs(channel_name: Optional[str] = None) -> dict[str, list[str]]:
    """Return channel-scoped pair IDs.

    Args:
        channel_name: Optional channel name to filter by.
    """
    config = load_pairings_config()
    if channel_name is None:
        return config.pairs_by_channel

    normalized_channel_name = channel_name.strip()
    return {normalized_channel_name: config.pairs_by_channel.get(normalized_channel_name, [])}


def get_pairs_for_channel(channel_name: str) -> list[str]:
    """Return the pair IDs configured for a specific channel config name."""
    normalized_channel_name = channel_name.strip()
    if not normalized_channel_name:
        raise ValueError("channel_name must not be empty or whitespace")

    config = load_pairings_config()
    return config.pairs_by_channel.get(normalized_channel_name, []).copy()


def add_pair(channel_name: str, pair_id: str) -> Path:
    """Add a pair ID to a channel's pairing allowlist."""
    normalized_channel_name = channel_name.strip()
    normalized_pair_id = pair_id.strip()
    if not normalized_channel_name:
        raise ValueError("channel_name must not be empty or whitespace")
    if not normalized_pair_id:
        raise ValueError("pair_id must not be empty or whitespace")

    config = load_pairings_config()
    existing = config.pairs_by_channel.get(normalized_channel_name, [])
    if normalized_pair_id not in existing:
        config.pairs_by_channel[normalized_channel_name] = [*existing, normalized_pair_id]
    return save_pairings_config(config)


def delete_pair(channel_name: str, pair_id: str) -> Path:
    """Delete a pair ID from a channel's pairing allowlist.

    Raises:
        FileNotFoundError: If the channel/pair entry does not exist.
    """
    normalized_channel_name = channel_name.strip()
    normalized_pair_id = pair_id.strip()
    if not normalized_channel_name:
        raise ValueError("channel_name must not be empty or whitespace")
    if not normalized_pair_id:
        raise ValueError("pair_id must not be empty or whitespace")

    config = load_pairings_config()
    existing = config.pairs_by_channel.get(normalized_channel_name, [])
    if normalized_pair_id not in existing:
        raise FileNotFoundError(
            f"No pair '{normalized_pair_id}' found for channel '{normalized_channel_name}'"
        )

    updated = [value for value in existing if value != normalized_pair_id]
    if updated:
        config.pairs_by_channel[normalized_channel_name] = updated
    else:
        config.pairs_by_channel.pop(normalized_channel_name, None)

    return save_pairings_config(config)
