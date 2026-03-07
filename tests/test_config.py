"""Tests for config models and loader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_expose.config.models import ExposureConfig, ProviderConfig, TelegramClientConfig
from llm_expose.config.loader import (
    delete_config,
    list_configs,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_valid_config(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        assert cfg.provider_name == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.api_key is None
        assert cfg.base_url is None

    def test_strips_whitespace(self) -> None:
        cfg = ProviderConfig(provider_name="  openai  ", model=" gpt-4o ")
        assert cfg.provider_name == "openai"
        assert cfg.model == "gpt-4o"

    def test_empty_provider_name_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(provider_name="", model="gpt-4o")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(provider_name="openai", model="")

    def test_whitespace_only_provider_name_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(provider_name="   ", model="gpt-4o")

    def test_temperature_boundaries(self) -> None:
        cfg_low = ProviderConfig(provider_name="openai", model="gpt-4o", temperature=0.0)
        cfg_high = ProviderConfig(provider_name="openai", model="gpt-4o", temperature=2.0)
        assert cfg_low.temperature == 0.0
        assert cfg_high.temperature == 2.0

    def test_temperature_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(provider_name="openai", model="gpt-4o", temperature=3.0)

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(provider_name="openai", model="gpt-4o", max_tokens=0)

    def test_local_config_with_base_url(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )
        assert cfg.base_url == "http://localhost:1234/v1"


class TestTelegramClientConfig:
    def test_valid_config(self) -> None:
        cfg = TelegramClientConfig(bot_token="123456:ABC-DEF")
        assert cfg.client_type == "telegram"
        assert cfg.bot_token == "123456:ABC-DEF"

    def test_strips_whitespace_from_token(self) -> None:
        cfg = TelegramClientConfig(bot_token="  123456:ABC  ")
        assert cfg.bot_token == "123456:ABC"

    def test_empty_token_raises(self) -> None:
        with pytest.raises(Exception):
            TelegramClientConfig(bot_token="")

    def test_whitespace_only_token_raises(self) -> None:
        with pytest.raises(Exception):
            TelegramClientConfig(bot_token="   ")


class TestExposureConfig:
    def _make_config(self, name: str = "test") -> ExposureConfig:
        return ExposureConfig(
            name=name,
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )

    def test_valid_config(self) -> None:
        cfg = self._make_config("my-exposure")
        assert cfg.name == "my-exposure"

    def test_name_strips_whitespace(self) -> None:
        cfg = self._make_config("  hello  ")
        assert cfg.name == "hello"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_config("")

    def test_name_with_slash_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_config("bad/name")

    def test_name_with_backslash_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_config("bad\\name")


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for config storage."""
    return tmp_path / "configs"


class TestConfigLoader:
    def _make_exposure(self, name: str = "test-exposure") -> ExposureConfig:
        return ExposureConfig(
            name=name,
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )

    def test_save_and_load_roundtrip(self, tmp_config_dir: Path) -> None:
        cfg = self._make_exposure("my-bot")
        saved_path = save_config(cfg, config_dir=tmp_config_dir)
        assert saved_path.exists()

        loaded = load_config("my-bot", config_dir=tmp_config_dir)
        assert loaded.name == cfg.name
        assert loaded.provider.provider_name == cfg.provider.provider_name
        assert loaded.provider.model == cfg.provider.model
        assert loaded.client.bot_token == cfg.client.bot_token

    def test_load_nonexistent_raises(self, tmp_config_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("does-not-exist", config_dir=tmp_config_dir)

    def test_list_empty_dir(self, tmp_config_dir: Path) -> None:
        result = list_configs(config_dir=tmp_config_dir)
        assert result == []

    def test_list_returns_all_names(self, tmp_config_dir: Path) -> None:
        for name in ("charlie", "alpha", "beta"):
            save_config(self._make_exposure(name), config_dir=tmp_config_dir)
        result = list_configs(config_dir=tmp_config_dir)
        assert result == ["alpha", "beta", "charlie"]  # sorted

    def test_delete_config(self, tmp_config_dir: Path) -> None:
        cfg = self._make_exposure("to-delete")
        save_config(cfg, config_dir=tmp_config_dir)
        delete_config("to-delete", config_dir=tmp_config_dir)
        assert list_configs(config_dir=tmp_config_dir) == []

    def test_delete_nonexistent_raises(self, tmp_config_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            delete_config("ghost", config_dir=tmp_config_dir)

    def test_get_config_dir_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from llm_expose.config.loader import get_config_dir
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        assert get_config_dir() == tmp_path

    def test_saved_config_is_valid_yaml(self, tmp_config_dir: Path) -> None:
        import yaml
        cfg = self._make_exposure("yaml-check")
        saved_path = save_config(cfg, config_dir=tmp_config_dir)
        with saved_path.open() as fh:
            data = yaml.safe_load(fh)
        assert data["name"] == "yaml-check"
        assert data["provider"]["provider_name"] == "openai"
