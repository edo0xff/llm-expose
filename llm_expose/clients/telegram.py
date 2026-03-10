"""Telegram client adapter using python-telegram-bot."""

from __future__ import annotations

import asyncio
import base64
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from llm_expose.clients.base import BaseClient, MessageHandler as LLMHandler, MessageResponse
from llm_expose.config.models import TelegramClientConfig
from llm_expose.core.content_parts import build_user_content

logger = logging.getLogger(__name__)
MARKDOWN_PARSE_MODE = "Markdown"


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

    async def _reply_text_safe(self, message, text: str, **kwargs) -> None:
        """Send a reply using Markdown; retry as plain text on parse errors."""
        try:
            await message.reply_text(text, parse_mode=MARKDOWN_PARSE_MODE, **kwargs)
        except BadRequest as exc:
            if "Can't parse entities" not in str(exc):
                raise
            logger.warning("Markdown parse failed in reply_text, retrying plain text: %s", exc)
            await message.reply_text(text, **kwargs)

    async def _edit_message_text_safe(self, query, text: str, **kwargs) -> None:
        """Edit a message using Markdown; retry as plain text on parse errors."""
        try:
            await query.edit_message_text(text, parse_mode=MARKDOWN_PARSE_MODE, **kwargs)
        except BadRequest as exc:
            if "Can't parse entities" not in str(exc):
                raise
            logger.warning(
                "Markdown parse failed in edit_message_text, retrying plain text: %s",
                exc,
            )
            await query.edit_message_text(text, **kwargs)

    async def _send_message_safe(self, bot, chat_id: str, text: str, **kwargs) -> None:
        """Send a message using Markdown; retry as plain text on parse errors."""
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=MARKDOWN_PARSE_MODE,
                **kwargs,
            )
        except BadRequest as exc:
            if "Can't parse entities" not in str(exc):
                raise
            logger.warning("Markdown parse failed in send_message, retrying plain text: %s", exc)
            await bot.send_message(chat_id=chat_id, text=text, **kwargs)

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        if update.message:
            await update.message.reply_text(
                "👋 Hello! I'm a telegram bot powered by an LLM. Let's setup some configurations!"
            )
            await update.message.reply_text(
                "1) Let me know how should I call you? Please reply with your name or nickname."
            )
            await update.message.reply_text(
                "2) Second, please tell me how I should behave? You can specify a system prompt to guide my responses. For example, you can say 'You are a helpful assistant that provides concise answers.' Or something simplier like 'Be friendly and use emojis!'"
            )
            await update.message.reply_text(
                "3) Finally, you can specify any additional instructions or preferences for our interactions. For example, you can say 'Quiero que me respondas en español' or any other instructions!"
            )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages by delegating to the LLM handler."""
        if not update.message:
            return

        user_text = update.message.text or update.message.caption or ""
        image_urls = await self._extract_image_data_urls(update, context)
        if not user_text and not image_urls:
            return

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
            # Keep backward compatibility with one-argument handlers used in
            # tests/custom integrations, but pass channel context when the
            # orchestrator handler is registered.
            bound_self = getattr(self._handler, "__self__", None)
            if bound_self is not None and bound_self.__class__.__name__ == "Orchestrator":
                message_content = build_user_content(user_text, image_urls=image_urls)
                reply = await self._handler(
                    chat_id,
                    user_text,
                    message_content=message_content,
                )
            else:
                reply = await self._handler(user_text)
        except Exception as exc:
            logger.exception("Error from LLM handler: %s", exc)
            reply = "⚠️ Sorry, I encountered an error. Please try again."

        # Check if reply is a structured MessageResponse with approval metadata
        if isinstance(reply, MessageResponse):
            if reply.approval_id:
                # Create inline keyboard with Approve/Reject buttons
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "✅ Approve",
                            callback_data=f"approve:{reply.approval_id}"
                        ),
                        InlineKeyboardButton(
                            "❌ Reject",
                            callback_data=f"reject:{reply.approval_id}"
                        ),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await self._reply_text_safe(
                    update.message,
                    reply.content,
                    reply_markup=reply_markup,
                )
            else:
                # No approval needed, just send the content
                await self._reply_text_safe(
                    update.message,
                    reply.content,
                )
        else:
            # Plain string response (backward compatibility)
            await self._reply_text_safe(update.message, reply)

    async def _extract_image_data_urls(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> list[str]:
        """Extract Telegram photo attachments as data URLs."""
        if not update.message or not update.message.photo:
            return []

        image_urls: list[str] = []
        # Telegram provides sizes from smallest to largest.
        best_photo = update.message.photo[-1]
        try:
            telegram_file = await context.bot.get_file(best_photo.file_id)
            payload: bytes | None = None

            download_as_bytearray = getattr(telegram_file, "download_as_bytearray", None)
            if callable(download_as_bytearray):
                payload = bytes(await download_as_bytearray())

            if payload:
                encoded = base64.b64encode(payload).decode("ascii")
                image_urls.append(f"data:image/jpeg;base64,{encoded}")
        except Exception as exc:
            logger.warning("Failed to extract photo attachment: %s", exc)

        return image_urls

    async def _handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle button press callbacks for approval decisions."""
        if not update.callback_query:
            return

        query = update.callback_query
        chat_id = str(query.message.chat.id) if query.message else None
        
        if not chat_id or not query.data:
            await query.answer("Invalid request.")
            return

        # Parse callback data: format is "approve:approval_id" or "reject:approval_id"
        try:
            decision, approval_id = query.data.split(":", 1)
        except ValueError:
            await query.answer("Invalid callback data.")
            return

        if decision not in ("approve", "reject"):
            await query.answer("Unknown action.")
            return

        # Answer the callback query immediately to remove button loading state
        await query.answer("Processing...")

        # Format as text command and send to orchestrator
        command_text = f"{decision} {approval_id}"
        logger.info(
            "Button press from user %s in chat %s: %s",
            update.effective_user,
            chat_id,
            command_text,
        )

        try:
            bound_self = getattr(self._handler, "__self__", None)
            if bound_self is not None and bound_self.__class__.__name__ == "Orchestrator":
                reply = await self._handler(chat_id, command_text)
            else:
                reply = await self._handler(command_text)
        except Exception as exc:
            logger.exception("Error from LLM handler during callback: %s", exc)
            reply = "⚠️ Sorry, I encountered an error processing your decision."

        # Extract content if reply is MessageResponse
        reply_text = reply.content if isinstance(reply, MessageResponse) else reply

        # Edit the original message to show the decision and result
        decision_emoji = "✅" if decision == "approve" else "❌"
        decision_text = "approved" if decision == "approve" else "rejected"
        status_message = f"{decision_emoji} Tool execution: {decision_text}\n\n{reply_text}"
        
        if query.message:
            try:
                await self._edit_message_text_safe(
                    query,
                    status_message,
                )
            except Exception as exc:
                # If editing fails (e.g., message too old), send a new message
                logger.warning("Could not edit message: %s", exc)
                await self._send_message_safe(context.bot, chat_id, status_message)

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
            MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_handler(CallbackQueryHandler(self._handle_callback_query))

        logger.info("Starting Telegram bot (polling)…")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot is running. Press Ctrl+C to stop.")

        # Keep running until stop() signals shutdown.
        await self._stop_event.wait()

    async def send_message(self, user_id: str, text: str) -> dict:
        """Send a direct message to a specific user.

        Uses the Markdown parsing with fallback to plain text on parse errors,
        consistent with reply and edit methods.

        Args:
            user_id: Telegram chat_id as string.
            text: Message text to send.

        Returns:
            Dict with keys: message_id, timestamp, status, user_id.

        Raises:
            RuntimeError: If the Telegram app cannot be initialized.
            BadRequest: If send fails (invalid chat_id, permissions, etc.).
        """
        from datetime import datetime as dt

        # Initialize the app if not already done
        if self._app is None:
            self._app = Application.builder().token(self._config.bot_token).build()
            await self._app.initialize()

        try:
            # Try to send with Markdown formatting first
            try:
                message = await self._app.bot.send_message(
                    chat_id=user_id,
                    text=text,
                    parse_mode=MARKDOWN_PARSE_MODE,
                )
            except BadRequest as exc:
                if "Can't parse entities" not in str(exc):
                    raise
                # Retry without Markdown on parse error
                logger.warning("Markdown parse failed in send_message, retrying plain text: %s", exc)
                message = await self._app.bot.send_message(chat_id=user_id, text=text)
            
            return {
                "message_id": str(message.message_id),
                "timestamp": dt.utcnow().isoformat() + "Z",
                "status": "sent",
                "user_id": user_id,
            }
        except BadRequest as exc:
            logger.error(
                "Failed to send message to user %s: %s",
                user_id,
                exc,
            )
            raise

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
