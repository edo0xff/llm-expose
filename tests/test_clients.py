"""Tests for the Telegram client adapter and orchestrator."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from llm_expose.clients.base import BaseClient, MessageResponse
from llm_expose.clients.telegram import TelegramClient
from llm_expose.config.models import (
    ExposureConfig,
    MCPConfig,
    MCPServerConfig,
    MCPSettingsConfig,
    ProviderConfig,
    TelegramClientConfig,
)
from llm_expose.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# BaseClient
# ---------------------------------------------------------------------------


class TestBaseClientInterface:
    def test_base_client_is_abstract(self) -> None:
        """BaseClient cannot be instantiated directly."""

        async def dummy(msg: str) -> str:
            return msg

        with pytest.raises(TypeError):
            BaseClient(handler=dummy)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# TelegramClient
# ---------------------------------------------------------------------------


class TestTelegramClientInit:
    def _make_client(self) -> TelegramClient:
        cfg = TelegramClientConfig(bot_token="123:tok")
        return TelegramClient(cfg, handler=AsyncMock(return_value="reply"))

    def test_is_base_client(self) -> None:
        client = self._make_client()
        assert isinstance(client, BaseClient)

    def test_config_stored(self) -> None:
        cfg = TelegramClientConfig(bot_token="999:abc")
        client = TelegramClient(cfg, handler=AsyncMock())
        assert client._config is cfg

    def test_app_initially_none(self) -> None:
        client = self._make_client()
        assert client._app is None


class TestTelegramClientHandlers:
    def _make_update(self, text: str = "Hello") -> MagicMock:
        """Build a minimal mock Telegram Update."""
        update = MagicMock()
        update.message.text = text
        update.message.chat.id = 42
        update.effective_user = "user42"
        return update

    @pytest.mark.asyncio
    async def test_handle_start_sends_reply(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        client = TelegramClient(cfg, handler=AsyncMock())

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await client._handle_start(update, context)
        assert update.message.reply_text.await_count >= 1
        # First reply should contain a greeting
        first_call = update.message.reply_text.await_args_list[0]
        assert len(first_call.args) == 1 and isinstance(first_call.args[0], str)

    @pytest.mark.asyncio
    async def test_handle_message_calls_handler(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        handler = AsyncMock(return_value="LLM reply")
        client = TelegramClient(cfg, handler=handler)

        update = self._make_update("What is 2+2?")
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await client._handle_message(update, context)

        handler.assert_awaited_once_with("What is 2+2?")
        update.message.reply_text.assert_called_once_with(
            "LLM reply", parse_mode="Markdown"
        )

    @pytest.mark.asyncio
    async def test_handle_message_retries_plain_text_on_markdown_parse_error(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        handler = AsyncMock(return_value="broken *markdown")
        client = TelegramClient(cfg, handler=handler)

        update = self._make_update("format this")
        update.message.reply_text = AsyncMock(
            side_effect=[
                BadRequest("Can't parse entities: can't find end of the entity"),
                None,
            ]
        )
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await client._handle_message(update, context)

        assert update.message.reply_text.await_count == 2
        first_call = update.message.reply_text.await_args_list[0]
        second_call = update.message.reply_text.await_args_list[1]
        assert first_call.args == ("broken *markdown",)
        assert first_call.kwargs == {"parse_mode": "Markdown"}
        assert second_call.args == ("broken *markdown",)
        assert second_call.kwargs == {}

    @pytest.mark.asyncio
    async def test_handle_message_returns_error_on_exception(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")

        async def failing_handler(msg: str) -> str:
            raise RuntimeError("LLM exploded")

        client = TelegramClient(cfg, handler=failing_handler)

        update = self._make_update("Boom")
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await client._handle_message(update, context)

        # Should reply with an error message, not raise
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "error" in reply_text.lower() or "⚠️" in reply_text

    @pytest.mark.asyncio
    async def test_handle_message_ignores_none_message(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        handler = AsyncMock()
        client = TelegramClient(cfg, handler=handler)

        update = MagicMock()
        update.message = None
        context = MagicMock()

        # Should not raise and not call handler
        await client._handle_message(update, context)
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_when_app_is_none_is_safe(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        client = TelegramClient(cfg, handler=AsyncMock())
        # Should not raise
        await client.stop()

    @pytest.mark.asyncio
    async def test_handle_message_forwards_photo_as_image_content(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")

        class Orchestrator:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            async def _handle_message(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "ok"

        orchestrator = Orchestrator()
        client = TelegramClient(cfg, handler=orchestrator._handle_message)

        update = MagicMock()
        update.message.text = "Look"
        update.message.caption = None
        update.message.chat.id = 42
        update.message.reply_text = AsyncMock()
        update.effective_user = "user42"
        photo = MagicMock()
        photo.file_id = "photo123"
        update.message.photo = [photo]

        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"jpg"))
        context.bot.get_file = AsyncMock(return_value=telegram_file)

        await client._handle_message(update, context)

        assert len(orchestrator.calls) == 1
        args, kwargs = orchestrator.calls[0]
        assert args[0] == "42"
        assert args[1] == "Look"
        message_content = kwargs["message_content"]
        assert isinstance(message_content, list)
        assert message_content[0]["type"] == "text"
        assert message_content[1]["type"] == "image_url"
        assert message_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_handle_message_sends_reference_image_from_structured_reply(self) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        handler = AsyncMock(
            return_value=MessageResponse(
                content="LLM reply",
                images=["https://example.com/reference.jpg"],
            )
        )
        client = TelegramClient(cfg, handler=handler)

        update = self._make_update("Analyze this")
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_photo = AsyncMock(return_value=MagicMock(message_id=99))

        await client._handle_message(update, context)

        update.message.reply_text.assert_called_once_with("LLM reply", parse_mode="Markdown")
        context.bot.send_photo.assert_awaited_once_with(
            chat_id="42",
            photo="https://example.com/reference.jpg",
        )

    @pytest.mark.asyncio
    async def test_send_file_sends_document(self, tmp_path) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        client = TelegramClient(cfg, handler=AsyncMock())

        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"pdf-content")

        message = MagicMock()
        message.message_id = 77
        message.document = MagicMock(file_id="telegram-file-1")

        client._app = MagicMock()
        client._app.bot.send_document = AsyncMock(return_value=message)

        result = await client.send_file("42", str(file_path))

        client._app.bot.send_document.assert_awaited_once()
        send_kwargs = client._app.bot.send_document.await_args.kwargs
        assert send_kwargs["chat_id"] == "42"
        assert result["status"] == "sent"
        assert result["user_id"] == "42"
        assert result["message_id"] == "77"
        assert result["file_name"] == "report.pdf"
        assert result["file_id"] == "telegram-file-1"

    @pytest.mark.asyncio
    async def test_send_file_raises_when_missing(self, tmp_path) -> None:
        cfg = TelegramClientConfig(bot_token="123:tok")
        client = TelegramClient(cfg, handler=AsyncMock())

        missing_path = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError):
            await client.send_file("42", str(missing_path))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def _make_orchestrator(self) -> tuple[Orchestrator, MagicMock, MagicMock]:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="The answer is 42.")
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()):
            orch = Orchestrator(config=config, provider=provider, client=client)
        return orch, provider, client

    @pytest.mark.asyncio
    async def test_handle_message_calls_provider(self) -> None:
        orch, provider, _ = self._make_orchestrator()
        reply = await orch._handle_message("What is the answer?")
        provider.complete.assert_awaited_once()
        assert reply == "The answer is 42."

    @pytest.mark.asyncio
    async def test_handle_message_echoes_first_image_as_reference(self) -> None:
        orch, _, _ = self._make_orchestrator()
        reply = await orch._handle_message(
            "42",
            "Analyze image",
            message_content=[
                {"type": "text", "text": "Analyze image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/one.jpg"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/two.jpg"}},
            ],
        )

        assert isinstance(reply, MessageResponse)
        assert reply.content == "The answer is 42."
        assert reply.images == ["https://example.com/one.jpg"]

    @pytest.mark.asyncio
    async def test_handle_message_blocks_unpaired_channel(self) -> None:
        config = ExposureConfig(
            name="test",
            channel_name="telegram-main",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="The answer is 42.")
        client = MagicMock()

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()), patch(
            "llm_expose.core.orchestrator.get_pairs_for_channel", return_value=["100"]
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)
            reply = await orch._handle_message("42", "Hello")

        assert reply == "This instance is not paired. Run llm-expose add pair 42"
        provider.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_message_recognizes_pair_added_without_restart(self) -> None:
        config = ExposureConfig(
            name="test",
            channel_name="telegram-main",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="The answer is 42.")
        client = MagicMock()

        channel_pairs: list[str] = []

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()), patch(
            "llm_expose.core.orchestrator.get_pairs_for_channel",
            side_effect=lambda channel_name: list(channel_pairs),
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

            blocked = await orch._handle_message("42", "Hello")
            assert blocked == "This instance is not paired. Run llm-expose add pair 42"
            provider.complete.assert_not_awaited()

            channel_pairs.append("42")

            allowed = await orch._handle_message("42", "Hello again")
            assert allowed == "The answer is 42."
            provider.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_message_blocks_pair_removed_without_restart(self) -> None:
        config = ExposureConfig(
            name="test",
            channel_name="telegram-main",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="The answer is 42.")
        client = MagicMock()

        channel_pairs = ["42"]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()), patch(
            "llm_expose.core.orchestrator.get_pairs_for_channel",
            side_effect=lambda channel_name: list(channel_pairs),
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

            allowed = await orch._handle_message("42", "Hello")
            assert allowed == "The answer is 42."
            provider.complete.assert_awaited_once()

            channel_pairs.clear()

            blocked = await orch._handle_message("42", "Hello after removal")
            assert blocked == "This instance is not paired. Run llm-expose add pair 42"
            assert provider.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_handle_message_allows_paired_channel(self) -> None:
        config = ExposureConfig(
            name="test",
            channel_name="telegram-main",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok"),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="The answer is 42.")
        client = MagicMock()

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()), patch(
            "llm_expose.core.orchestrator.get_pairs_for_channel", return_value=["42"]
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)
            reply = await orch._handle_message("42", "Hello")

        assert reply == "The answer is 42."
        provider.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_history_grows_with_turns(self) -> None:
        orch, provider, _ = self._make_orchestrator()
        # Initial history has only the system message
        assert len(orch._history) == 1

        await orch._handle_message("Turn 1")
        assert len(orch._history) == 3  # system + user + assistant

        await orch._handle_message("Turn 2")
        assert len(orch._history) == 5

    def test_orchestrator_uses_channel_system_prompt(self) -> None:
        import tempfile
        from pathlib import Path
        
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("You are specialized for this channel.")
            temp_prompt_path = f.name
        
        try:
            config = ExposureConfig(
                name="test",
                provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
                client=TelegramClientConfig(
                    bot_token="123:tok",
                    system_prompt_path=temp_prompt_path,
                ),
            )
            provider = MagicMock()
            client = MagicMock()

            with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()):
                orch = Orchestrator(config=config, provider=provider, client=client)

            assert orch._history[0]["content"] == "You are specialized for this channel."
        finally:
            # Clean up temporary file
            Path(temp_prompt_path).unlink()

    def test_orchestrator_uses_default_prompt_when_channel_prompt_missing(self) -> None:
        orch, _, _ = self._make_orchestrator()
        assert "helpful AI assistant" in str(orch._history[0]["content"])

    @pytest.mark.asyncio
    async def test_run_starts_client(self) -> None:
        orch, _, client = self._make_orchestrator()
        await orch.run()
        client.start.assert_awaited_once()
        client.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_sets_client_handler(self) -> None:
        orch, _, client = self._make_orchestrator()
        # Track which handler is passed to set_handler
        captured: list = []
        client.set_handler = lambda h: captured.append(h)
        await orch.run()
        # The orchestrator must have wired its _handle_message via set_handler.
        # Compare __func__ and __self__ because bound method objects are not singletons.
        assert len(captured) == 1
        assert captured[0].__func__ is Orchestrator._handle_message
        assert captured[0].__self__ is orch

    @pytest.mark.asyncio
    async def test_run_stops_client_when_start_fails(self) -> None:
        orch, _, client = self._make_orchestrator()
        client.start = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await orch.run()

        client.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_message_passes_mcp_tools_to_provider(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["remote-mcp"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="MCP reply")
        provider.complete_with_message = AsyncMock(
            side_effect=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mcp_server__search_docs",
                                "arguments": '{"query":"2+2"}',
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "MCP reply"},
            ]
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="optional"),
            servers=[
                MCPServerConfig(
                    name="remote-mcp",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    allowed_tools=["mcp_server__search_docs"],
                    enabled=True,
                )
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value="tool result")
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_server__search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ), patch("llm_expose.core.orchestrator.get_pairs_for_channel", return_value=["42"]):
            orch = Orchestrator(config=config, provider=provider, client=client)

        reply = await orch._handle_message("Use MCP")
        assert reply == "MCP reply"
        fake_runtime.initialize.assert_awaited_once()
        fake_runtime.execute_tool_call.assert_awaited_once()
        assert provider.complete_with_message.await_count == 2

    @pytest.mark.asyncio
    async def test_handle_message_passes_execution_context_to_runtime(self) -> None:
        config = ExposureConfig(
            name="test",
            channel_name="support",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["builtin-core"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="fallback")
        provider.complete_with_message = AsyncMock(
            side_effect=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_ctx",
                            "type": "function",
                            "function": {
                                "name": "llm_expose_get_invocation_context",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "done"},
            ]
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="optional"),
            servers=[
                MCPServerConfig(
                    name="builtin-core",
                    transport="builtin",
                    allowed_tools=["llm_expose_get_invocation_context"],
                    enabled=True,
                )
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value=json.dumps({"channel_id": "42"}))
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_context",
                    "description": "Get invocation context",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ), patch("llm_expose.core.orchestrator.get_pairs_for_channel", return_value=["42"]):
            orch = Orchestrator(config=config, provider=provider, client=client)

            reply = await orch._handle_message(
                "42",
                "Use context",
                message_context={
                    "platform": "telegram",
                    "chat_type": "private",
                    "effective_user_id": "42",
                },
            )

        assert reply == "done"
        _, kwargs = fake_runtime.execute_tool_call.await_args
        execution_context = kwargs["execution_context"]
        assert execution_context.channel_id == "42"
        assert execution_context.channel_name == "support"
        assert execution_context.subject_kind == "user"
        assert execution_context.initiator_user_id == "42"
        assert execution_context.platform == "telegram"

    @pytest.mark.asyncio
    async def test_handle_message_skips_disabled_or_invalid_mcp_servers(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["missing-url"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="No tools reply")
        provider.complete_with_message = AsyncMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_required",
                        "type": "function",
                        "function": {
                            "name": "mcp_server__search_docs",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="required"),
            servers=[
                MCPServerConfig(
                    name="disabled",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    enabled=False,
                ),
                MCPServerConfig(
                    name="missing-url",
                    transport="stdio",
                    command="npx",
                    enabled=True,
                ),
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value="tool result")
        fake_runtime.get_tool_server_name = MagicMock(return_value="missing-url")
        fake_runtime.get_server_config = MagicMock(return_value=MCPServerConfig(
            name="missing-url",
            transport="stdio",
            command="npx",
            tool_confirmation="default",
        ))
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_server__search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

        reply = await orch._handle_message("Hello")
        assert isinstance(reply, MessageResponse)
        assert "requires confirmation" in reply.content
        provider.complete.assert_not_awaited()
        fake_runtime.execute_tool_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_required_mode_approve_executes_pending_tool_calls(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["remote-mcp"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="fallback")
        provider.complete_with_message = AsyncMock(
            side_effect=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "mcp_server__search_docs",
                                "arguments": '{"query":"weather"}',
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "approved final"},
            ]
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="required"),
            servers=[
                MCPServerConfig(
                    name="remote-mcp",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    allowed_tools=["mcp_server__search_docs"],
                    enabled=True,
                )
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value="tool result")
        fake_runtime.get_tool_server_name = MagicMock(return_value="remote-mcp")
        fake_runtime.get_server_config = MagicMock(return_value=MCPServerConfig(
            name="remote-mcp",
            transport="sse",
            url="https://mcp.example.com/sse",
            tool_confirmation="default",
        ))
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_server__search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

        prompt = await orch._handle_message("42", "Use MCP")
        assert isinstance(prompt, MessageResponse)
        assert "requires confirmation" in prompt.content
        assert prompt.approval_id
        approval_id = prompt.approval_id

        final = await orch._handle_message("42", f"approve {approval_id}")
        assert final == "approved final"
        fake_runtime.execute_tool_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_required_mode_reject_skips_tool_execution(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["remote-mcp"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="rejection handled")
        provider.complete_with_message = AsyncMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "mcp_server__search_docs",
                            "arguments": '{"query":"news"}',
                        },
                    }
                ],
            }
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="required"),
            servers=[
                MCPServerConfig(
                    name="remote-mcp",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    allowed_tools=["mcp_server__search_docs"],
                    enabled=True,
                )
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value="tool result")
        fake_runtime.get_tool_server_name = MagicMock(return_value="remote-mcp")
        fake_runtime.get_server_config = MagicMock(return_value=MCPServerConfig(
            name="remote-mcp",
            transport="sse",
            url="https://mcp.example.com/sse",
            tool_confirmation="default",
        ))
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_server__search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

        prompt = await orch._handle_message("42", "Use MCP")
        assert isinstance(prompt, MessageResponse)
        assert prompt.approval_id
        approval_id = prompt.approval_id

        final = await orch._handle_message("42", f"reject {approval_id}")
        assert final == "rejection handled"
        fake_runtime.execute_tool_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pending_approval_blocks_new_messages_until_decision(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["remote-mcp"]),
        )
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="fallback")
        provider.complete_with_message = AsyncMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {"name": "mcp_server__search_docs", "arguments": "{}"},
                    }
                ],
            }
        )
        client = MagicMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        mcp_config = MCPConfig(
            settings=MCPSettingsConfig(confirmation_mode="required"),
            servers=[
                MCPServerConfig(
                    name="remote-mcp",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    enabled=True,
                )
            ],
        )
        fake_runtime = MagicMock()
        fake_runtime.initialize = AsyncMock()
        fake_runtime.shutdown = AsyncMock()
        fake_runtime.execute_tool_call = AsyncMock(return_value="tool result")
        fake_runtime.get_tool_server_name = MagicMock(return_value="remote-mcp")
        fake_runtime.get_server_config = MagicMock(return_value=MCPServerConfig(
            name="remote-mcp",
            transport="sse",
            url="https://mcp.example.com/sse",
            tool_confirmation="default",
        ))
        fake_runtime.tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_server__search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager", return_value=fake_runtime
        ):
            orch = Orchestrator(config=config, provider=provider, client=client)

        await orch._handle_message("42", "Use MCP")
        blocked = await orch._handle_message("42", "another question")
        assert "waiting for confirmation" in blocked

    def test_orchestrator_does_not_create_runtime_without_channel_attachments(self) -> None:
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=[]),
        )
        provider = MagicMock()
        client = MagicMock()
        mcp_config = MCPConfig(
            servers=[
                MCPServerConfig(
                    name="remote-mcp",
                    transport="sse",
                    url="https://mcp.example.com/sse",
                    enabled=True,
                )
            ]
        )

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=mcp_config), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager"
        ) as runtime_cls:
            orch = Orchestrator(config=config, provider=provider, client=client)

        assert orch._mcp_runtime is None
        runtime_cls.assert_not_called()

    def test_orchestrator_creates_runtime_for_injected_builtin_attachment(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["builtin-core"]),
        )
        provider = MagicMock()
        client = MagicMock()
        fake_runtime = MagicMock()

        with patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager",
            return_value=fake_runtime,
        ) as runtime_cls:
            orch = Orchestrator(config=config, provider=provider, client=client)

        assert orch._mcp_runtime is fake_runtime
        runtime_cls.assert_called_once()
        runtime_config = runtime_cls.call_args.args[0]
        assert [server.name for server in runtime_config.servers] == ["builtin-core"]
        assert runtime_config.servers[0].transport == "builtin"

    def test_orchestrator_warns_for_missing_attached_mcp_servers(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="llm_expose.core.orchestrator")
        config = ExposureConfig(
            name="test",
            provider=ProviderConfig(provider_name="openai", model="gpt-4o"),
            client=TelegramClientConfig(bot_token="123:tok", mcp_servers=["ghost"]),
        )
        provider = MagicMock()
        client = MagicMock()

        with patch("llm_expose.core.orchestrator.load_mcp_config", return_value=MCPConfig()), patch(
            "llm_expose.core.orchestrator.MCPRuntimeManager"
        ) as runtime_cls:
            orch = Orchestrator(config=config, provider=provider, client=client)

        assert orch._mcp_runtime is None
        runtime_cls.assert_not_called()
        assert "missing MCP server attachments" in caplog.text
