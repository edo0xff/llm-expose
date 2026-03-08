"""Configuration loading and saving utilities for llm-expose."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from llm_expose.config.models import ProviderConfig, TelegramClientConfig

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
