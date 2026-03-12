"""Tests for CLI helpers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from pathlib import Path

import pytest
import typer

from llm_expose.config.models import MCPConfig, MCPSettingsConfig, ProviderConfig, TelegramClientConfig
from llm_expose.cli.main import (
    add_pair_cmd,
    add_channel,
    add_mcp_cmd,
    add_model,
    delete_model_cmd,
    delete_channel_cmd,
    delete_pair_cmd,
    delete_mcp_cmd,
    list_pairs_cmd,
    message,
    _parse_multi_select_numbers,
    _select_mcp_servers_for_channel,
)


class TestCliHelpers:
    def test_parse_multi_select_numbers(self) -> None:
        assert _parse_multi_select_numbers("1,2,3") == [1, 2, 3]
        assert _parse_multi_select_numbers(" 1 , 2 , 2 , 3 ") == [1, 2, 3]
        assert _parse_multi_select_numbers("") is None
        assert _parse_multi_select_numbers("a,2") is None

    def test_select_mcp_servers_select_all(self) -> None:
        with patch("llm_expose.cli.main.Prompt.ask", return_value="4"):
            selected = _select_mcp_servers_for_channel(["foo", "bar", "baz"])
        assert selected == ["foo", "bar", "baz"]

    def test_select_mcp_servers_select_none(self) -> None:
        with patch("llm_expose.cli.main.Prompt.ask", return_value="5"):
            selected = _select_mcp_servers_for_channel(["foo", "bar", "baz"])
        assert selected == []

    def test_select_mcp_servers_select_subset(self) -> None:
        with patch("llm_expose.cli.main.Prompt.ask", return_value="1,3"):
            selected = _select_mcp_servers_for_channel(["foo", "bar", "baz"])
        assert selected == ["foo", "baz"]

    def test_select_mcp_servers_empty_options(self) -> None:
        selected = _select_mcp_servers_for_channel([])
        assert selected == []

    def test_add_channel_saves_without_custom_system_prompt(self) -> None:
        with patch("llm_expose.cli.main._print_banner"), patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main._select_from_list", return_value="Telegram"), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=["mcp-a"]
        ), patch(
            "llm_expose.cli.main._select_mcp_servers_for_channel", return_value=["mcp-a"]
        ), patch(
            "llm_expose.cli.main.Prompt.ask", side_effect=["my-channel", "123:tok"]
        ), patch(
            "llm_expose.cli.main.Confirm.ask", return_value=False
        ), patch(
            "llm_expose.cli.main.save_channel", return_value=Path("/tmp/ch.yaml")
        ) as save_channel_mock:
            add_channel(name=None, bot_token=None, model_name=None, mcp_server=[], system_prompt_path=None, yes=False, no_input=False)

        saved_cfg = save_channel_mock.call_args.args[1]
        assert saved_cfg.bot_token == "123:tok"
        assert saved_cfg.mcp_servers == ["mcp-a"]
        assert saved_cfg.system_prompt_path is None

    def test_add_channel_saves_selected_builtin_server(self) -> None:
        with patch("llm_expose.cli.main._print_banner"), patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main._select_from_list", return_value="Telegram"), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=["builtin-core", "mcp-a"]
        ), patch(
            "llm_expose.cli.main._select_mcp_servers_for_channel", return_value=["builtin-core"]
        ), patch(
            "llm_expose.cli.main.Prompt.ask", side_effect=["my-channel", "123:tok"]
        ), patch(
            "llm_expose.cli.main.Confirm.ask", return_value=False
        ), patch(
            "llm_expose.cli.main.save_channel", return_value=Path("/tmp/ch.yaml")
        ) as save_channel_mock:
            add_channel(name=None, bot_token=None, model_name=None, mcp_server=[], system_prompt_path=None, yes=False, no_input=False)

        saved_cfg = save_channel_mock.call_args.args[1]
        assert saved_cfg.mcp_servers == ["builtin-core"]

    def test_add_channel_saves_with_custom_system_prompt(self) -> None:
        with patch("llm_expose.cli.main._print_banner"), patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main._select_from_list", return_value="Telegram"), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch(
            "llm_expose.cli.main._select_mcp_servers_for_channel", return_value=[]
        ), patch(
            "llm_expose.cli.main.Prompt.ask", side_effect=["my-channel", "123:tok", "/tmp/prompt.txt"]
        ), patch(
            "llm_expose.cli.main.Confirm.ask", return_value=True
        ), patch(
            "builtins.open", create=True
        ) as open_mock, patch(
            "llm_expose.cli.main.save_channel", return_value=Path("/tmp/ch.yaml")
        ) as save_channel_mock:
            open_mock.return_value.__enter__.return_value.read.return_value = "Channel prompt"
            add_channel(name=None, bot_token=None, model_name=None, mcp_server=[], system_prompt_path=None, yes=False, no_input=False)

        saved_cfg = save_channel_mock.call_args.args[1]
        assert saved_cfg.system_prompt_path == "/tmp/prompt.txt"

    def test_add_pair_cmd_with_args(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.add_channel_pair"
        ) as add_pair_mock:
            add_pair_cmd("42", channel="telegram-main", no_input=False)

        add_pair_mock.assert_called_once_with("telegram-main", "42")

    def test_add_pair_cmd_prompts_for_pair_id_when_missing(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.Prompt.ask", return_value="84"
        ), patch("llm_expose.cli.main.add_channel_pair") as add_pair_mock:
            add_pair_cmd(None, channel="telegram-main", no_input=False)

        add_pair_mock.assert_called_once_with("telegram-main", "84")

    def test_add_mcp_cmd_saves_stdio_server(self) -> None:
        with patch("llm_expose.cli.main._print_banner"), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch(
            "llm_expose.cli.main._select_from_list",
            side_effect=["stdio", "default"],
        ), patch(
            "llm_expose.cli.main.Prompt.ask",
            side_effect=["stdio-core", "uvx", "mcp-server --flag"],
        ), patch(
            "llm_expose.cli.main.Confirm.ask", return_value=True
        ), patch(
            "llm_expose.cli.main.save_mcp_server", return_value=Path("/tmp/mcp.yaml")
        ) as save_mcp_server_mock:
            add_mcp_cmd(name=None, transport=None, command=None, args=None, url=None, enabled=None, tool_confirmation=None, yes=False, no_input=False)

        saved_cfg = save_mcp_server_mock.call_args.args[0]
        assert saved_cfg.name == "stdio-core"
        assert saved_cfg.transport == "stdio"
        assert saved_cfg.command == "uvx"
        assert saved_cfg.args == ["mcp-server", "--flag"]
        assert saved_cfg.url is None

    def test_add_mcp_cmd_saves_sse_server(self) -> None:
        with patch("llm_expose.cli.main._print_banner"), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch(
            "llm_expose.cli.main._select_from_list",
            side_effect=["sse", "required"],
        ), patch(
            "llm_expose.cli.main.Prompt.ask",
            side_effect=["sse-core", "http://localhost:3000/sse"],
        ), patch(
            "llm_expose.cli.main.Confirm.ask", return_value=True
        ), patch(
            "llm_expose.cli.main.save_mcp_server", return_value=Path("/tmp/mcp.yaml")
        ) as save_mcp_server_mock:
            add_mcp_cmd(name=None, transport=None, command=None, args=None, url=None, enabled=None, tool_confirmation=None, yes=False, no_input=False)

        saved_cfg = save_mcp_server_mock.call_args.args[0]
        assert saved_cfg.name == "sse-core"
        assert saved_cfg.transport == "sse"
        assert saved_cfg.command is None
        assert saved_cfg.args == []
        assert saved_cfg.url == "http://localhost:3000/sse"

    def test_list_pairs_cmd_shows_configured_pairs(self) -> None:
        with patch("llm_expose.cli.main.list_pairs", return_value={"telegram-main": ["42"]}), patch(
            "llm_expose.cli.main.console.print"
        ) as print_mock:
            list_pairs_cmd(channel=None)

        assert print_mock.called

    def test_delete_pair_cmd_with_args(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.delete_channel_pair"
        ) as delete_pair_mock:
            delete_pair_cmd("42", channel="telegram-main", yes=False, no_input=False)

        delete_pair_mock.assert_called_once_with("telegram-main", "42")

    def test_delete_pair_cmd_selects_pair_when_missing(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["42", "84"]
        ), patch("llm_expose.cli.main._select_from_list", return_value="84"), patch(
            "llm_expose.cli.main.delete_channel_pair"
        ) as delete_pair_mock:
            delete_pair_cmd(None, channel="telegram-main", yes=False, no_input=False)

        delete_pair_mock.assert_called_once_with("telegram-main", "84")

    def test_message_suppress_send_requires_llm_completion(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.console.print"
        ) as print_mock:
            message(
                channel="ops",
                user_id="12345",
                text="draft a reply",
                llm_completion=False,
                suppress_send=True,
                system_prompt_file=None,
                image=[],
            )

        assert exc_info.value.exit_code == 1
        print_mock.assert_called_once_with(
            "[red]Error: --suppress-send requires --llm-completion.[/red]"
        )

    def test_message_suppress_send_returns_llm_json_without_delivery(self) -> None:
        client_cfg = TelegramClientConfig(
            bot_token="123:tok",
            model_name="ops-model",
        )
        provider_cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o-mini",
            api_key="secret",
            base_url=None,
        )

        with patch("llm_expose.cli.main.load_channel", return_value=client_cfg), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["12345"]
        ), patch("llm_expose.cli.main.load_model", return_value=provider_cfg), patch(
            "llm_expose.cli.main.LiteLLMProvider"
        ) as provider_cls_mock, patch(
            "llm_expose.cli.main.TelegramClient"
        ) as client_cls_mock, patch(
            "llm_expose.cli.main.asyncio.run", return_value="Generated reply"
        ) as asyncio_run_mock, patch(
            "llm_expose.cli.main.console.print"
        ) as print_mock:
            message(
                channel="ops",
                user_id="12345",
                text="Draft a reply",
                llm_completion=True,
                suppress_send=True,
                system_prompt_file=None,
                image=[],
            )

        provider_cls_mock.assert_called_once_with(provider_cfg)
        client_cls_mock.assert_not_called()
        assert asyncio_run_mock.call_count == 1

        result = json.loads(print_mock.call_args.args[0])
        assert result["status"] == "suppressed"
        assert result["channel"] == "ops"
        assert result["user_id"] == "12345"
        assert result["llm_response"] == "Generated reply"
        assert result["llm_model"] == "ops-model"
        assert "message_id" not in result

    def test_message_without_suppress_send_still_delivers(self) -> None:
        client_cfg = TelegramClientConfig(
            bot_token="123:tok",
            model_name="ops-model",
        )
        provider_cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o-mini",
            api_key="secret",
            base_url=None,
        )

        with patch("llm_expose.cli.main.load_channel", return_value=client_cfg), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["12345"]
        ), patch("llm_expose.cli.main.load_model", return_value=provider_cfg), patch(
            "llm_expose.cli.main.LiteLLMProvider"
        ), patch("llm_expose.cli.main.TelegramClient") as client_cls_mock, patch(
            "llm_expose.cli.main.asyncio.run",
            side_effect=[
                "Generated reply",
                {
                    "message_id": "99",
                    "timestamp": "2026-03-10T00:00:00Z",
                    "status": "sent",
                    "user_id": "12345",
                },
            ],
        ) as asyncio_run_mock, patch("llm_expose.cli.main.console.print") as print_mock:
            message(
                channel="ops",
                user_id="12345",
                text="Draft a reply",
                llm_completion=True,
                suppress_send=False,
                system_prompt_file=None,
                image=[],
            )

        client_cls_mock.assert_called_once()
        assert asyncio_run_mock.call_count == 2

        result = json.loads(print_mock.call_args.args[0])
        assert result["status"] == "sent"
        assert result["message_id"] == "99"
        assert result["llm_response"] == "Generated reply"
        assert result["llm_model"] == "ops-model"

    def test_message_with_file_sends_text_and_file(self) -> None:
        client_cfg = TelegramClientConfig(bot_token="123:tok")

        with patch("llm_expose.cli.main.load_channel", return_value=client_cfg), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["12345"]
        ), patch("llm_expose.cli.main.TelegramClient") as client_cls_mock, patch(
            "llm_expose.cli.main.asyncio.run",
            return_value={
                "message_id": "99",
                "timestamp": "2026-03-10T00:00:00Z",
                "status": "sent",
                "user_id": "12345",
                "file_reference": {
                    "message_id": "100",
                    "timestamp": "2026-03-10T00:00:01Z",
                    "status": "sent",
                    "user_id": "12345",
                    "file_name": "report.pdf",
                },
            },
        ) as asyncio_run_mock, patch("llm_expose.cli.main.console.print") as print_mock, patch(
            "llm_expose.cli.main.Path"
        ) as path_mock:
            path_instance = path_mock.return_value.expanduser.return_value
            path_instance.exists.return_value = True
            path_instance.is_file.return_value = True
            path_instance.__str__.return_value = "C:/tmp/report.pdf"

            message(
                channel="ops",
                user_id="12345",
                text="Some file is here:",
                llm_completion=False,
                suppress_send=False,
                system_prompt_file=None,
                image=[],
                file="C:/tmp/report.pdf",
            )

        client_cls_mock.assert_called_once()
        assert asyncio_run_mock.call_count == 1

        result = json.loads(print_mock.call_args.args[0])
        assert result["status"] == "sent"
        assert result["message_id"] == "99"
        assert result["file_reference"]["message_id"] == "100"

    def test_message_tool_aware_completion_suppress_send_keeps_sender_for_tools(self) -> None:
        client_cfg = TelegramClientConfig(
            bot_token="123:tok",
            model_name="ops-model",
            mcp_servers=["builtin-core"],
        )
        provider_cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o-mini",
            api_key="secret",
            base_url=None,
        )

        handler_mock = AsyncMock()
        handler_mock.complete = AsyncMock(return_value="Generated reply")

        with patch("llm_expose.cli.main.load_channel", return_value=client_cfg), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["12345"]
        ), patch("llm_expose.cli.main.load_model", return_value=provider_cfg), patch(
            "llm_expose.cli.main.load_mcp_config", return_value=MCPConfig()
        ), patch("llm_expose.cli.main.LiteLLMProvider"), patch(
            "llm_expose.cli.main.TelegramClient"
        ) as client_cls_mock, patch(
            "llm_expose.cli.main.ToolAwareCompletion"
        ) as completion_cls_mock, patch(
            "llm_expose.cli.main.console.print"
        ):
            completion_cls_mock.return_value.__aenter__.return_value = handler_mock
            completion_cls_mock.return_value.__aexit__.return_value = None

            message(
                channel="ops",
                user_id="12345",
                text="Draft a reply",
                llm_completion=True,
                suppress_send=True,
                system_prompt_file=None,
                image=[],
            )

        client_cls_mock.assert_called_once()
        execution_context = handler_mock.complete.await_args.kwargs["execution_context"]
        assert execution_context.sender is client_cls_mock.return_value

    def test_message_tool_aware_completion_includes_redacted_attachment_descriptor(self, tmp_path) -> None:
        image_path = tmp_path / "camera.jpg"
        image_path.write_bytes(b"abc")

        client_cfg = TelegramClientConfig(
            bot_token="123:tok",
            model_name="ops-model",
            mcp_servers=["builtin-core"],
        )
        provider_cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o-mini",
            api_key="secret",
            base_url=None,
        )

        handler_mock = AsyncMock()
        handler_mock.complete = AsyncMock(return_value="Generated reply")

        with patch("llm_expose.cli.main.load_channel", return_value=client_cfg), patch(
            "llm_expose.cli.main.get_pairs_for_channel", return_value=["12345"]
        ), patch("llm_expose.cli.main.load_model", return_value=provider_cfg), patch(
            "llm_expose.cli.main.load_mcp_config",
            return_value=MCPConfig(settings=MCPSettingsConfig(expose_attachment_paths=False)),
        ), patch("llm_expose.cli.main.LiteLLMProvider"), patch(
            "llm_expose.cli.main.ToolAwareCompletion"
        ) as completion_cls_mock, patch("llm_expose.cli.main.console.print"):
            completion_cls_mock.return_value.__aenter__.return_value = handler_mock
            completion_cls_mock.return_value.__aexit__.return_value = None

            message(
                channel="ops",
                user_id="12345",
                text="Draft a reply",
                llm_completion=True,
                suppress_send=True,
                system_prompt_file=None,
                image=[str(image_path)],
            )

        execution_context = handler_mock.complete.await_args.kwargs["execution_context"]
        assert len(execution_context.attachments) == 1
        assert execution_context.attachments[0]["filename"] == "camera.jpg"
        assert execution_context.attachments[0]["path"] is None
        attachment_ref = execution_context.attachments[0]["attachment_ref"]
        assert isinstance(attachment_ref, str)
        assert attachment_ref.startswith("att_")
        assert execution_context.attachment_paths_by_ref[attachment_ref] == str(image_path.resolve())

    def test_message_file_not_found_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.console.print"
        ) as print_mock, patch("llm_expose.cli.main.Path") as path_mock:
            path_instance = path_mock.return_value.expanduser.return_value
            path_instance.exists.return_value = False
            path_instance.is_file.return_value = False

            message(
                channel="ops",
                user_id="12345",
                text="Some file is here:",
                llm_completion=False,
                suppress_send=False,
                system_prompt_file=None,
                image=[],
                file="C:/tmp/missing.pdf",
            )

        assert exc_info.value.exit_code == 1
        print_mock.assert_called_once_with("[red]Error: File not found: C:/tmp/missing.pdf[/red]")

    def test_message_file_conflicts_with_llm_completion(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.console.print"
        ) as print_mock:
            message(
                channel="ops",
                user_id="12345",
                text="Draft this",
                llm_completion=True,
                suppress_send=False,
                system_prompt_file=None,
                image=[],
                file="C:/tmp/report.pdf",
            )

        assert exc_info.value.exit_code == 1
        print_mock.assert_called_once_with(
            "[red]Error: --file cannot be used with --llm-completion.[/red]"
        )


# ---------------------------------------------------------------------------
# Headless mode tests
# ---------------------------------------------------------------------------

class TestHeadlessAddModel:
    """Test add_model() with -y / --no-input flags."""

    def test_headless_creates_model_with_all_flags(self) -> None:
        with patch("llm_expose.cli.main.list_models", return_value=[]), patch(
            "llm_expose.cli.main.save_model", return_value=Path("/tmp/m.yaml")
        ) as save_mock:
            add_model(
                name="my-model",
                provider="openai",
                model_id="gpt-4o",
                base_url=None,
                api_key="sk-secret",
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[1]
        assert saved_cfg.provider_name == "openai"
        assert saved_cfg.model == "gpt-4o"
        assert saved_cfg.api_key == "sk-secret"

    def test_headless_local_model_no_api_key_required(self) -> None:
        with patch("llm_expose.cli.main.list_models", return_value=[]), patch(
            "llm_expose.cli.main.save_model", return_value=Path("/tmp/m.yaml")
        ) as save_mock:
            add_model(
                name="local-model",
                provider="local",
                model_id="llama3",
                base_url="http://localhost:11434",
                api_key=None,
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[1]
        assert saved_cfg.provider_name == "local"
        assert saved_cfg.base_url == "http://localhost:11434"

    def test_headless_fails_when_no_input_without_required_flags(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_models", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_model(
                name=None,  # missing → should fail fast
                provider="openai",
                model_id="gpt-4o",
                base_url=None,
                api_key="sk-secret",
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_when_no_input_missing_model_id(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_models", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_model(
                name="my-model",
                provider="openai",
                model_id=None,
                base_url=None,
                api_key="sk-secret",
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_when_existing_name_without_yes(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_models", return_value=["existing-model"]
        ), patch("llm_expose.cli.main.console.print"):
            add_model(
                name="existing-model",
                provider="openai",
                model_id="gpt-4o",
                base_url=None,
                api_key="sk-secret",
                yes=False,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_overwrites_existing_model_with_yes(self) -> None:
        with patch("llm_expose.cli.main.list_models", return_value=["existing-model"]), patch(
            "llm_expose.cli.main.save_model", return_value=Path("/tmp/m.yaml")
        ) as save_mock:
            add_model(
                name="existing-model",
                provider="openai",
                model_id="gpt-4o-mini",
                base_url=None,
                api_key="sk-secret",
                yes=True,
                no_input=True,
            )

        assert save_mock.called


class TestHeadlessAddChannel:
    """Test add_channel() with -y / --no-input flags."""

    def test_headless_creates_channel_with_all_flags(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=[]), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=["mcp-a"]
        ), patch("llm_expose.cli.main.save_channel", return_value=Path("/tmp/ch.yaml")) as save_mock:
            add_channel(
                name="my-channel",
                bot_token="123:tok",
                model_name="my-model",
                mcp_server=["mcp-a"],
                system_prompt_path=None,
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[1]
        assert saved_cfg.bot_token == "123:tok"
        assert saved_cfg.model_name == "my-model"
        assert saved_cfg.mcp_servers == ["mcp-a"]

    def test_headless_fails_when_mcp_server_not_found(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main.list_mcp_servers", return_value=[]), patch(
            "llm_expose.cli.main.console.print"
        ):
            add_channel(
                name="my-channel",
                bot_token="123:tok",
                model_name="my-model",
                mcp_server=["nonexistent"],
                system_prompt_path=None,
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_without_name(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_channel(
                name=None,
                bot_token="123:tok",
                model_name="my-model",
                mcp_server=[],
                system_prompt_path=None,
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_without_bot_token(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_channel(
                name="my-channel",
                bot_token=None,
                model_name="my-model",
                mcp_server=[],
                system_prompt_path=None,
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_creates_channel_without_model(self) -> None:
        """model_name is optional even in headless mode."""
        with patch("llm_expose.cli.main.list_channels", return_value=[]), patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch("llm_expose.cli.main.save_channel", return_value=Path("/tmp/ch.yaml")) as save_mock:
            add_channel(
                name="my-channel",
                bot_token="123:tok",
                model_name=None,
                mcp_server=[],
                system_prompt_path=None,
                yes=True,
                no_input=True,
            )

        assert save_mock.called


class TestHeadlessAddMcp:
    """Test add_mcp_cmd() with -y / --no-input flags."""

    def test_headless_creates_stdio_server(self) -> None:
        with patch("llm_expose.cli.main.list_mcp_servers", return_value=[]), patch(
            "llm_expose.cli.main.save_mcp_server", return_value=Path("/tmp/mcp.yaml")
        ) as save_mock:
            add_mcp_cmd(
                name="my-server",
                transport="stdio",
                command="uv",
                args=["run", "mcp-server"],
                url=None,
                enabled=True,
                tool_confirmation="default",
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[0]
        assert saved_cfg.name == "my-server"
        assert saved_cfg.transport == "stdio"
        assert saved_cfg.command == "uv"
        assert saved_cfg.args == ["run", "mcp-server"]

    def test_headless_creates_sse_server(self) -> None:
        with patch("llm_expose.cli.main.list_mcp_servers", return_value=[]), patch(
            "llm_expose.cli.main.save_mcp_server", return_value=Path("/tmp/mcp.yaml")
        ) as save_mock:
            add_mcp_cmd(
                name="my-sse",
                transport="sse",
                command=None,
                args=None,
                url="http://localhost:3000/sse",
                enabled=True,
                tool_confirmation="required",
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[0]
        assert saved_cfg.transport == "sse"
        assert saved_cfg.url == "http://localhost:3000/sse"
        assert saved_cfg.command is None

    def test_headless_fails_when_stdio_missing_command(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_mcp_cmd(
                name="my-server",
                transport="stdio",
                command=None,
                args=None,
                url=None,
                enabled=True,
                tool_confirmation="default",
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_when_sse_missing_url(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_mcp_cmd(
                name="my-sse",
                transport="sse",
                command=None,
                args=None,
                url=None,
                enabled=True,
                tool_confirmation="default",
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_fails_without_name(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=[]
        ), patch("llm_expose.cli.main.console.print"):
            add_mcp_cmd(
                name=None,
                transport="stdio",
                command="uv",
                args=None,
                url=None,
                enabled=True,
                tool_confirmation="default",
                yes=True,
                no_input=True,
            )

        assert exc_info.value.exit_code == 1

    def test_headless_defaults_enabled_and_tool_confirmation(self) -> None:
        with patch("llm_expose.cli.main.list_mcp_servers", return_value=[]), patch(
            "llm_expose.cli.main.save_mcp_server", return_value=Path("/tmp/mcp.yaml")
        ) as save_mock:
            add_mcp_cmd(
                name="my-server",
                transport="stdio",
                command="uv",
                args=None,
                url=None,
                enabled=None,    # should default to True
                tool_confirmation=None,  # should default to "default"
                yes=True,
                no_input=True,
            )

        saved_cfg = save_mock.call_args.args[0]
        assert saved_cfg.enabled is True
        assert saved_cfg.tool_confirmation == "default"


class TestHeadlessAddPair:
    """Test add_pair_cmd() with --no-input flag."""

    def test_headless_fails_without_channel(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=["telegram-main"]
        ), patch("llm_expose.cli.main.console.print"):
            add_pair_cmd("42", channel=None, no_input=True)

        assert exc_info.value.exit_code == 1

    def test_headless_with_channel_adds_pair(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.add_channel_pair"
        ) as add_pair_mock:
            add_pair_cmd("42", channel="telegram-main", no_input=True)

        add_pair_mock.assert_called_once_with("telegram-main", "42")


class TestHeadlessDeleteModel:
    """Test delete_model_cmd() with -y / --no-input flags."""

    def test_headless_deletes_model_with_yes_flag(self) -> None:
        with patch("llm_expose.cli.main.list_models", return_value=["my-model"]), patch(
            "llm_expose.cli.main.delete_model"
        ) as delete_mock, patch("llm_expose.cli.main.console.print"):
            delete_model_cmd("my-model", yes=True, no_input=True)

        delete_mock.assert_called_once_with("my-model")

    def test_headless_fails_without_yes(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_models", return_value=["my-model"]
        ), patch("llm_expose.cli.main.console.print"):
            delete_model_cmd("my-model", yes=False, no_input=True)

        assert exc_info.value.exit_code == 1

    def test_headless_fails_when_model_not_found(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_models", return_value=["other-model"]
        ), patch("llm_expose.cli.main.console.print"):
            delete_model_cmd("ghost-model", yes=True, no_input=True)

        assert exc_info.value.exit_code == 1


class TestHeadlessDeleteChannel:
    """Test delete_channel_cmd() with -y / --no-input flags."""

    def test_headless_deletes_channel_with_yes_flag(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["my-channel"]), patch(
            "llm_expose.cli.main.delete_channel"
        ) as delete_mock, patch("llm_expose.cli.main.console.print"):
            delete_channel_cmd("my-channel", yes=True, no_input=True)

        delete_mock.assert_called_once_with("my-channel")

    def test_headless_fails_without_yes(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=["my-channel"]
        ), patch("llm_expose.cli.main.console.print"):
            delete_channel_cmd("my-channel", yes=False, no_input=True)

        assert exc_info.value.exit_code == 1


class TestHeadlessDeleteMcp:
    """Test delete_mcp_cmd() with -y / --no-input flags."""

    def test_headless_deletes_server_with_yes_flag(self) -> None:
        with patch("llm_expose.cli.main.list_mcp_servers", return_value=["my-server"]), patch(
            "llm_expose.cli.main.delete_mcp_server"
        ) as delete_mock, patch("llm_expose.cli.main.console.print"):
            delete_mcp_cmd("my-server", yes=True, no_input=True)

        delete_mock.assert_called_once_with("my-server")

    def test_headless_fails_without_yes(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_mcp_servers", return_value=["my-server"]
        ), patch("llm_expose.cli.main.console.print"):
            delete_mcp_cmd("my-server", yes=False, no_input=True)

        assert exc_info.value.exit_code == 1


class TestHeadlessDeletePair:
    """Test delete_pair_cmd() with --no-input flag."""

    def test_headless_fails_without_channel(self) -> None:
        with pytest.raises(typer.Exit) as exc_info, patch(
            "llm_expose.cli.main.list_channels", return_value=["telegram-main"]
        ), patch("llm_expose.cli.main.console.print"):
            delete_pair_cmd("42", channel=None, no_input=True)

        assert exc_info.value.exit_code == 1

    def test_headless_deletes_pair_with_channel(self) -> None:
        with patch("llm_expose.cli.main.list_channels", return_value=["telegram-main"]), patch(
            "llm_expose.cli.main.delete_channel_pair"
        ) as delete_mock:
            delete_pair_cmd("42", channel="telegram-main", no_input=True)

        delete_mock.assert_called_once_with("telegram-main", "42")
