"""Tests for CLI helpers."""

from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

from llm_expose.cli.main import (
    add_channel,
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
            add_channel()

        saved_cfg = save_channel_mock.call_args.args[1]
        assert saved_cfg.bot_token == "123:tok"
        assert saved_cfg.mcp_servers == ["mcp-a"]
        assert saved_cfg.system_prompt is None

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
            add_channel()

        saved_cfg = save_channel_mock.call_args.args[1]
        assert saved_cfg.system_prompt == "Channel prompt"
