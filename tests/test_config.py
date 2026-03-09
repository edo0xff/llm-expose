"""Tests for config models and loader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_expose.config.models import (
    MCPServerConfig,
    MCPSettingsConfig,
    ProviderConfig,
    TelegramClientConfig,
)
from llm_expose.config.loader import (
    delete_model,
    delete_channel,
    delete_mcp_server,
    get_mcp_server,
    list_models,
    list_channels,
    list_mcp_servers,
    load_model,
    load_channel,
    load_mcp_settings,
    save_model,
    save_channel,
    save_mcp_server,
    save_mcp_settings,
)


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_valid_config(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        assert cfg.provider_name == "openai"
        assert cfg.model == "gpt-4o"
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


class TestMCPConfig:
    def test_valid_stdio_server(self) -> None:
        cfg = MCPServerConfig(name="filesystem", transport="stdio", command="npx")
        assert cfg.name == "filesystem"
        assert cfg.transport == "stdio"
        assert cfg.command == "npx"

    def test_valid_sse_server(self) -> None:
        cfg = MCPServerConfig(name="remote", transport="sse", url="http://localhost:3000/sse")
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:3000/sse"

    def test_settings_defaults(self) -> None:
        settings = MCPSettingsConfig()
        assert settings.confirmation_mode == "optional"
        assert settings.tool_timeout_seconds == 30

    def test_server_allows_tool_allowlist(self) -> None:
        cfg = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:3000/sse",
            allowed_tools=["search_docs"],
        )
        assert cfg.allowed_tools == ["search_docs"]


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestModelLoader:
    """Tests for model (provider) configuration storage."""

    def _make_provider(self) -> ProviderConfig:
        return ProviderConfig(provider_name="openai", model="gpt-4o")

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        saved_path = save_model("my-model", cfg)
        assert saved_path.exists()

        loaded = load_model("my-model")
        assert loaded.provider_name == cfg.provider_name
        assert loaded.model == cfg.model

    def test_load_nonexistent_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_model("does-not-exist")

    def test_list_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        result = list_models()
        assert result == []

    def test_list_returns_all_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        for name in ("charlie", "alpha", "beta"):
            save_model(name, cfg)
        result = list_models()
        assert result == ["alpha", "beta", "charlie"]  # sorted

    def test_delete_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        save_model("to-delete", cfg)
        delete_model("to-delete")
        assert list_models() == []

    def test_delete_nonexistent_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_model("ghost")


class TestChannelLoader:
    """Tests for channel (client) configuration storage."""

    def _make_client(self) -> TelegramClientConfig:
        return TelegramClientConfig(bot_token="123:tok")

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        saved_path = save_channel("my-channel", cfg)
        assert saved_path.exists()

        loaded = load_channel("my-channel")
        assert loaded.client_type == cfg.client_type
        assert loaded.bot_token == cfg.bot_token

    def test_load_nonexistent_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_channel("does-not-exist")

    def test_list_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        result = list_channels()
        assert result == []

    def test_list_returns_all_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        for name in ("charlie", "alpha", "beta"):
            save_channel(name, cfg)
        result = list_channels()
        assert result == ["alpha", "beta", "charlie"]  # sorted

    def test_delete_channel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        save_channel("to-delete", cfg)
        delete_channel("to-delete")
        assert list_channels() == []

    def test_delete_nonexistent_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_channel("ghost")


class TestMCPLoader:
    def test_save_and_list_servers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        save_mcp_server(MCPServerConfig(name="db", transport="sse", url="http://localhost:3000/sse"))
        assert list_mcp_servers() == ["db", "web"]

    def test_http_transport_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that HTTP transport is properly supported in MCP server config."""
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="http-server", transport="http", url="http://localhost:8080/mcp"))
        server = get_mcp_server("http-server")
        assert server.transport == "http"
        assert server.url == "http://localhost:8080/mcp"
        assert list_mcp_servers() == ["http-server"]

    def test_get_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        server = get_mcp_server("web")
        assert server.command == "npx"

    def test_delete_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        delete_mcp_server("web")
        assert list_mcp_servers() == []

    def test_delete_nonexistent_server_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_mcp_server("ghost")

    def test_load_default_mcp_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        settings = load_mcp_settings()
        assert settings.confirmation_mode == "optional"
        assert settings.tool_timeout_seconds == 30

    def test_save_mcp_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_settings(MCPSettingsConfig(confirmation_mode="required", tool_timeout_seconds=45))
        settings = load_mcp_settings()
        assert settings.confirmation_mode == "required"
        assert settings.tool_timeout_seconds == 45
