"""Telegram client adapter using python-telegram-bot."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from llm_expose.clients.base import BaseClient, MessageHandler as LLMHandler
from llm_expose.config.models import TelegramClientConfig

logger = logging.getLogger(__name__)


class TelegramClient(BaseClient):
    """Messaging client adapter for Telegram.

    Listens for incoming text messages and commands via the Telegram Bot API
    (long-polling) and forwards them to the registered LLM handler.

    Args:
        config: Telegram-specific configuration (bot token).
        handler: Async callable that receives the user's text and returns the
            LLM's reply.
    """

    def __init__(self, config: TelegramClientConfig, handler: LLMHandler) -> None:
        super().__init__(handler)
        self._config = config
        self._app: Application | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Telegram update handlers
    # ------------------------------------------------------------------

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        if update.message:
            await update.message.reply_text(
                "👋 Hello! I'm powered by an LLM. Send me a message and I'll reply!"
            )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages by delegating to the LLM handler."""
        if not update.message or not update.message.text:
            return

        user_text = update.message.text
        chat_id = str(update.message.chat.id)
        logger.info(
            "Received message from user %s in chat %s",
            update.effective_user,
            chat_id,
        )

        # Show a typing action while waiting for the LLM
        if update.message.chat:
            await context.bot.send_chat_action(
                chat_id=update.message.chat.id, action="typing"
            )

        try:
            reply = await self._handler(chat_id, user_text)
        except Exception as exc:
            logger.exception("Error from LLM handler: %s", exc)
            reply = "⚠️ Sorry, I encountered an error. Please try again."

        await update.message.reply_text(reply, parse_mode="Markdown")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the Telegram application and start polling for updates."""
        self._stop_event = asyncio.Event()
        self._app = (
            Application.builder()
            .token(self._config.bot_token)
            .build()
        )

        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.info("Starting Telegram bot (polling)…")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot is running. Press Ctrl+C to stop.")

        # Keep running until stop() signals shutdown.
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Stop polling and shut down the Telegram application."""
        if self._app is None:
            return

        if self._stop_event is not None:
            self._stop_event.set()

        logger.info("Stopping Telegram bot…")
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None
        self._stop_event = None
