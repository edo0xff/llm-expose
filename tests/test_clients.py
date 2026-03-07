"""Tests for the Telegram client adapter and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_expose.clients.base import BaseClient
from llm_expose.clients.telegram import TelegramClient
from llm_expose.config.models import (
    ExposureConfig,
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
        update.message.reply_text.assert_called_once()
        # Should contain a greeting
        args = update.message.reply_text.call_args[0]
        assert len(args) == 1 and isinstance(args[0], str)

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
        update.message.reply_text.assert_called_once_with("LLM reply")

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
        orch = Orchestrator(config=config, provider=provider, client=client)
        return orch, provider, client

    @pytest.mark.asyncio
    async def test_handle_message_calls_provider(self) -> None:
        orch, provider, _ = self._make_orchestrator()
        reply = await orch._handle_message("What is the answer?")
        provider.complete.assert_awaited_once()
        assert reply == "The answer is 42."

    @pytest.mark.asyncio
    async def test_history_grows_with_turns(self) -> None:
        orch, provider, _ = self._make_orchestrator()
        # Initial history has only the system message
        assert len(orch._history) == 1

        await orch._handle_message("Turn 1")
        assert len(orch._history) == 3  # system + user + assistant

        await orch._handle_message("Turn 2")
        assert len(orch._history) == 5

    @pytest.mark.asyncio
    async def test_run_starts_client(self) -> None:
        orch, _, client = self._make_orchestrator()
        await orch.run()
        client.start.assert_awaited_once()

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
