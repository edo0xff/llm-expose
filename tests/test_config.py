"""Tests for config models and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_expose.config.loader import (
    add_pair,
    delete_channel,
    delete_mcp_server,
    delete_model,
    delete_pair,
    get_mcp_server,
    get_pairs_for_channel,
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
from llm_expose.config.models import (
    ClientConfig,
    DiscordClientConfig,
    MCPServerConfig,
    MCPSettingsConfig,
    ProviderConfig,
    TelegramClientConfig,
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
        with pytest.raises(ValidationError):
            ProviderConfig(provider_name="", model="gpt-4o")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider_name="openai", model="")

    def test_whitespace_only_provider_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider_name="   ", model="gpt-4o")

    def test_local_config_with_base_url(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )
        assert cfg.base_url == "http://localhost:1234/v1"

    def test_supports_vision_override(self) -> None:
        cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            supports_vision=True,
        )
        assert cfg.supports_vision is True


class TestTelegramClientConfig:
    def test_valid_config(self) -> None:
        cfg = TelegramClientConfig(bot_token="123456:ABC-DEF")
        assert cfg.client_type == "telegram"
        assert cfg.bot_token == "123456:ABC-DEF"

    def test_strips_whitespace_from_token(self) -> None:
        cfg = TelegramClientConfig(bot_token="  123456:ABC  ")
        assert cfg.bot_token == "123456:ABC"

    def test_empty_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            TelegramClientConfig(bot_token="")

    def test_whitespace_only_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            TelegramClientConfig(bot_token="   ")

    def test_mcp_servers_defaults_to_empty(self) -> None:
        cfg = TelegramClientConfig(bot_token="123456:ABC")
        assert cfg.mcp_servers == []

    def test_mcp_servers_are_normalized_and_deduplicated(self) -> None:
        cfg = TelegramClientConfig(
            bot_token="123456:ABC",
            mcp_servers=["  foo ", "bar", "foo", "", "   ", "baz"],
        )
        assert cfg.mcp_servers == ["foo", "bar", "baz"]

    def test_system_prompt_path_strips_outer_whitespace(self) -> None:
        cfg = TelegramClientConfig(
            bot_token="123456:ABC",
            system_prompt_path="  /path/to/prompt.txt  ",
        )
        assert cfg.system_prompt_path == "/path/to/prompt.txt"

    def test_system_prompt_path_whitespace_only_becomes_none(self) -> None:
        cfg = TelegramClientConfig(
            bot_token="123456:ABC",
            system_prompt_path="   ",
        )
        assert cfg.system_prompt_path is None

    def test_system_prompt_path_accepts_any_path(self) -> None:
        """Test that system_prompt_path accepts arbitrary file paths."""
        cfg = TelegramClientConfig(
            bot_token="123456:ABC",
            system_prompt_path="/home/user/my_prompt.txt",
        )
        assert cfg.system_prompt_path == "/home/user/my_prompt.txt"

    def test_system_prompt_path_with_relative_path(self) -> None:
        """Test that system_prompt_path accepts relative paths."""
        cfg = TelegramClientConfig(
            bot_token="123456:ABC",
            system_prompt_path="./prompts/my_prompt.md",
        )
        assert cfg.system_prompt_path == "./prompts/my_prompt.md"


# ---------------------------------------------------------------------------
# DiscordClientConfig
# ---------------------------------------------------------------------------


class TestDiscordClientConfig:
    def test_valid_config(self) -> None:
        cfg = DiscordClientConfig(bot_token="MTk4NjIyNDgzNDcxOTI1MjQ4.deadbeef")
        assert cfg.client_type == "discord"
        assert cfg.bot_token == "MTk4NjIyNDgzNDcxOTI1MjQ4.deadbeef"

    def test_strips_whitespace_from_token(self) -> None:
        cfg = DiscordClientConfig(bot_token="  my-discord-token  ")
        assert cfg.bot_token == "my-discord-token"

    def test_empty_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            DiscordClientConfig(bot_token="")

    def test_whitespace_only_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            DiscordClientConfig(bot_token="   ")

    def test_mcp_servers_defaults_to_empty(self) -> None:
        cfg = DiscordClientConfig(bot_token="discord-tok")
        assert cfg.mcp_servers == []

    def test_mcp_servers_are_normalized_and_deduplicated(self) -> None:
        cfg = DiscordClientConfig(
            bot_token="discord-tok",
            mcp_servers=["  foo ", "bar", "foo", "", "   ", "baz"],
        )
        assert cfg.mcp_servers == ["foo", "bar", "baz"]

    def test_system_prompt_path_strips_whitespace(self) -> None:
        cfg = DiscordClientConfig(
            bot_token="discord-tok",
            system_prompt_path="  /path/prompt.txt  ",
        )
        assert cfg.system_prompt_path == "/path/prompt.txt"

    def test_system_prompt_path_whitespace_only_becomes_none(self) -> None:
        cfg = DiscordClientConfig(
            bot_token="discord-tok",
            system_prompt_path="   ",
        )
        assert cfg.system_prompt_path is None

    def test_model_name_strips_whitespace(self) -> None:
        cfg = DiscordClientConfig(bot_token="discord-tok", model_name="  gpt-4o  ")
        assert cfg.model_name == "gpt-4o"

    def test_model_name_whitespace_only_becomes_none(self) -> None:
        cfg = DiscordClientConfig(bot_token="discord-tok", model_name="   ")
        assert cfg.model_name is None

    def test_client_config_union_routes_to_discord(self) -> None:
        from pydantic import TypeAdapter

        adapter: TypeAdapter[ClientConfig] = TypeAdapter(ClientConfig)
        cfg = adapter.validate_python(
            {"client_type": "discord", "bot_token": "discord-tok"}
        )
        assert isinstance(cfg, DiscordClientConfig)
        assert cfg.client_type == "discord"

    def test_client_config_union_routes_to_telegram(self) -> None:
        from pydantic import TypeAdapter

        adapter: TypeAdapter[ClientConfig] = TypeAdapter(ClientConfig)
        cfg = adapter.validate_python(
            {"client_type": "telegram", "bot_token": "123:tok"}
        )
        assert isinstance(cfg, TelegramClientConfig)
        assert cfg.client_type == "telegram"


class TestMCPConfig:
    def test_valid_stdio_server(self) -> None:
        cfg = MCPServerConfig(name="filesystem", transport="stdio", command="npx")
        assert cfg.name == "filesystem"
        assert cfg.transport == "stdio"
        assert cfg.command == "npx"

    def test_valid_sse_server(self) -> None:
        cfg = MCPServerConfig(
            name="remote", transport="sse", url="http://localhost:3000/sse"
        )
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:3000/sse"

    def test_valid_builtin_server(self) -> None:
        cfg = MCPServerConfig(name="builtin-core", transport="builtin")
        assert cfg.transport == "builtin"

    def test_settings_defaults(self) -> None:
        settings = MCPSettingsConfig()
        assert settings.confirmation_mode == "optional"
        assert settings.tool_timeout_seconds == 30
        assert settings.expose_attachment_paths is False

    def test_server_allows_tool_allowlist(self) -> None:
        cfg = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:3000/sse",
            allowed_tools=["search_docs"],
        )
        assert cfg.allowed_tools == ["search_docs"]

    def test_tool_confirmation_defaults_to_default(self) -> None:
        cfg = MCPServerConfig(name="filesystem", transport="stdio", command="npx")
        assert cfg.tool_confirmation == "default"

    def test_tool_confirmation_accepts_required(self) -> None:
        cfg = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            tool_confirmation="required",
        )
        assert cfg.tool_confirmation == "required"

    def test_tool_confirmation_accepts_never(self) -> None:
        cfg = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            tool_confirmation="never",
        )
        assert cfg.tool_confirmation == "never"


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestModelLoader:
    """Tests for model (provider) configuration storage."""

    def _make_provider(self) -> ProviderConfig:
        return ProviderConfig(provider_name="openai", model="gpt-4o")

    def test_save_and_load_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        saved_path = save_model("my-model", cfg)
        assert saved_path.exists()

        loaded = load_model("my-model")
        assert loaded.provider_name == cfg.provider_name
        assert loaded.model == cfg.model

    def test_load_nonexistent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_model("does-not-exist")

    def test_list_empty_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        result = list_models()
        assert result == []

    def test_list_returns_all_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        for name in ("charlie", "alpha", "beta"):
            save_model(name, cfg)
        result = list_models()
        assert result == ["alpha", "beta", "charlie"]  # sorted

    def test_delete_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_provider()
        save_model("to-delete", cfg)
        delete_model("to-delete")
        assert list_models() == []

    def test_delete_nonexistent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_model("ghost")


class TestChannelLoader:
    """Tests for channel (client) configuration storage."""

    def _make_client(self) -> TelegramClientConfig:
        return TelegramClientConfig(bot_token="123:tok")

    def test_save_and_load_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        saved_path = save_channel("my-channel", cfg)
        assert saved_path.exists()

        loaded = load_channel("my-channel")
        assert loaded.client_type == cfg.client_type
        assert loaded.bot_token == cfg.bot_token

    def test_save_and_load_roundtrip_with_system_prompt_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = TelegramClientConfig(
            bot_token="123:tok",
            mcp_servers=["filesystem"],
            system_prompt_path="/etc/prompts/channel.txt",
        )

        save_channel("channel-with-prompt", cfg)
        loaded = load_channel("channel-with-prompt")

        assert loaded.bot_token == "123:tok"
        assert loaded.mcp_servers == ["filesystem"]
        assert loaded.system_prompt_path == "/etc/prompts/channel.txt"

    def test_load_nonexistent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_channel("does-not-exist")

    def test_list_empty_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        result = list_channels()
        assert result == []

    def test_list_returns_all_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        for name in ("charlie", "alpha", "beta"):
            save_channel(name, cfg)
        result = list_channels()
        assert result == ["alpha", "beta", "charlie"]  # sorted

    def test_delete_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = self._make_client()
        save_channel("to-delete", cfg)
        delete_channel("to-delete")
        assert list_channels() == []

    def test_delete_nonexistent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_channel("ghost")

    def test_load_legacy_channel_defaults_mcp_servers_to_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        channel_path = tmp_path / "channels" / "legacy.yaml"
        channel_path.parent.mkdir(parents=True, exist_ok=True)
        channel_path.write_text(
            "name: legacy\nclient_type: telegram\nbot_token: 123:tok\n",
            encoding="utf-8",
        )

        loaded = load_channel("legacy")
        assert loaded.bot_token == "123:tok"
        assert loaded.mcp_servers == []
        assert loaded.system_prompt_path is None

    def test_save_and_load_discord_roundtrip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        cfg = DiscordClientConfig(
            bot_token="discord-tok",
            mcp_servers=["my-mcp"],
            system_prompt_path="/prompts/bot.txt",
            model_name="gpt-4o",
        )

        save_channel("discord-chan", cfg)
        loaded = load_channel("discord-chan")

        assert isinstance(loaded, DiscordClientConfig)
        assert loaded.client_type == "discord"
        assert loaded.bot_token == "discord-tok"
        assert loaded.mcp_servers == ["my-mcp"]
        assert loaded.system_prompt_path == "/prompts/bot.txt"
        assert loaded.model_name == "gpt-4o"

    def test_mixed_telegram_and_discord_channels(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        tg_cfg = TelegramClientConfig(bot_token="123:tok")
        dc_cfg = DiscordClientConfig(bot_token="discord-tok")

        save_channel("tg-chan", tg_cfg)
        save_channel("dc-chan", dc_cfg)

        loaded_tg = load_channel("tg-chan")
        loaded_dc = load_channel("dc-chan")

        assert loaded_tg.client_type == "telegram"
        assert loaded_dc.client_type == "discord"


class TestMCPLoader:
    def test_load_mcp_config_injects_builtin_server_when_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))

        config = load_mcp_config()

        assert [server.name for server in config.servers] == ["builtin-core"]
        assert config.servers[0].transport == "builtin"

    def test_save_and_list_servers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        save_mcp_server(
            MCPServerConfig(name="db", transport="sse", url="http://localhost:3000/sse")
        )
        assert list_mcp_servers() == ["builtin-core", "db", "web"]

    def test_http_transport_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that HTTP transport is properly supported in MCP server config."""
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(
            MCPServerConfig(
                name="http-server", transport="http", url="http://localhost:8080/mcp"
            )
        )
        server = get_mcp_server("http-server")
        assert server.transport == "http"
        assert server.url == "http://localhost:8080/mcp"
        assert list_mcp_servers() == ["builtin-core", "http-server"]

    def test_builtin_transport_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="builtin-core", transport="builtin"))
        server = get_mcp_server("builtin-core")
        assert server.transport == "builtin"
        assert server.command is None
        assert server.url is None

    def test_persisted_builtin_server_overrides_injected_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(
            MCPServerConfig(
                name="builtin-core",
                transport="builtin",
                enabled=False,
                tool_confirmation="never",
            )
        )

        config = load_mcp_config()

        assert [server.name for server in config.servers] == ["builtin-core"]
        assert config.servers[0].enabled is False
        assert config.servers[0].tool_confirmation == "never"

    def test_get_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        server = get_mcp_server("web")
        assert server.command == "npx"

    def test_get_builtin_server_without_persisted_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))

        server = get_mcp_server("builtin-core")

        assert server.transport == "builtin"

    def test_delete_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_server(MCPServerConfig(name="web", transport="stdio", command="npx"))
        delete_mcp_server("web")
        assert list_mcp_servers() == ["builtin-core"]

    def test_delete_nonexistent_server_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            delete_mcp_server("ghost")

    def test_load_default_mcp_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        settings = load_mcp_settings()
        assert settings.confirmation_mode == "optional"
        assert settings.tool_timeout_seconds == 30
        assert settings.expose_attachment_paths is False

    def test_save_mcp_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        save_mcp_settings(
            MCPSettingsConfig(
                confirmation_mode="required",
                tool_timeout_seconds=45,
                expose_attachment_paths=True,
            )
        )
        settings = load_mcp_settings()
        assert settings.confirmation_mode == "required"
        assert settings.tool_timeout_seconds == 45
        assert settings.expose_attachment_paths is True


class TestPairLoader:
    def test_list_pairs_defaults_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        assert list_pairs() == {}

    def test_add_pair_and_get_pairs_for_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        add_pair("telegram-main", "42")
        add_pair("telegram-main", "84")
        add_pair("telegram-main", "42")  # duplicate ignored

        assert get_pairs_for_channel("telegram-main") == ["42", "84"]

    def test_list_pairs_can_filter_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        add_pair("alpha", "1")
        add_pair("beta", "2")

        assert list_pairs("alpha") == {"alpha": ["1"]}
        assert list_pairs("missing") == {"missing": []}

    def test_delete_pair_removes_channel_when_last_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        add_pair("telegram-main", "42")
        delete_pair("telegram-main", "42")
        assert list_pairs() == {}

    def test_delete_missing_pair_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        add_pair("telegram-main", "42")
        with pytest.raises(FileNotFoundError):
            delete_pair("telegram-main", "999")
