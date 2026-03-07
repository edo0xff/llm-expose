"""Configuration loading and saving utilities for llm-expose."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from llm_expose.config.models import ExposureConfig

# Default directory where configs are persisted
_DEFAULT_CONFIG_DIR = Path.home() / ".llm-expose" / "configs"


def get_config_dir() -> Path:
    """Return the directory used to store exposure configs.

    The directory is resolved from the ``LLM_EXPOSE_CONFIG_DIR`` environment
    variable when set, otherwise defaults to ``~/.llm-expose/configs``.
    """
    env_dir = os.environ.get("LLM_EXPOSE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_CONFIG_DIR


def _config_path(name: str, config_dir: Optional[Path] = None) -> Path:
    """Return the filesystem path for a named config file."""
    directory = config_dir if config_dir is not None else get_config_dir()
    return directory / f"{name}.yaml"


def save_config(config: ExposureConfig, config_dir: Optional[Path] = None) -> Path:
    """Persist *config* to a YAML file inside *config_dir*.

    Creates the config directory if it does not already exist.

    Args:
        config: The :class:`ExposureConfig` instance to save.
        config_dir: Optional override for the config directory.

    Returns:
        The :class:`~pathlib.Path` where the config was written.
    """
    directory = config_dir if config_dir is not None else get_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.name}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, allow_unicode=True, sort_keys=False)
    return path


def load_config(name: str, config_dir: Optional[Path] = None) -> ExposureConfig:
    """Load an :class:`ExposureConfig` from disk by *name*.

    Args:
        name: The name used when the config was saved.
        config_dir: Optional override for the config directory.

    Returns:
        The deserialized :class:`ExposureConfig`.

    Raises:
        FileNotFoundError: If no config with the given name exists.
        ValueError: If the YAML is malformed or validation fails.
    """
    path = _config_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(f"No configuration named '{name}' found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ExposureConfig.model_validate(data)


def list_configs(config_dir: Optional[Path] = None) -> list[str]:
    """Return names of all saved exposure configs.

    Args:
        config_dir: Optional override for the config directory.

    Returns:
        A sorted list of config names (file stems without the ``.yaml`` extension).
    """
    directory = config_dir if config_dir is not None else get_config_dir()
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def delete_config(name: str, config_dir: Optional[Path] = None) -> None:
    """Delete a saved config by *name*.

    Args:
        name: The name of the config to delete.
        config_dir: Optional override for the config directory.

    Raises:
        FileNotFoundError: If no config with the given name exists.
    """
    path = _config_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(f"No configuration named '{name}' found at {path}")
    path.unlink()
