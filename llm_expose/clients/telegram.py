"""Telegram client adapter using python-telegram-bot."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from llm_expose.clients.base import BaseClient, MessageHandler as LLMHandler, MessageResponse
from llm_expose.config.models import TelegramClientConfig
from llm_expose.core.content_parts import build_user_content

logger = logging.getLogger(__name__)

MARKDOWN_PARSE_MODE = "MarkdownV2"

RESERVED_PARSE_CHARACTERS = [
  '(',
  ')'
]

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
        self._approval_messages: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Telegram update handlers
    # ------------------------------------------------------------------

    async def _reply_text_safe(self, message, text: str, **kwargs):
        """Send a reply using Markdown; retry as plain text on parse errors."""
        try:
            # Escape special characters in Telegram markdown
            for char in RESERVED_PARSE_CHARACTERS:
                text = text.replace(char, f"\\{char}")
            return await message.reply_text(text, parse_mode=MARKDOWN_PARSE_MODE, **kwargs)
        except BadRequest as exc:
            if "Can't parse entities" not in str(exc):
                raise
            logger.warning("Markdown parse failed in reply_text, retrying plain text: %s", exc)
            return await message.reply_text(text, **kwargs)

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

    async def _edit_chat_message_text_safe(
        self,
        bot,
        chat_id: str,
        message_id: str,
        text: str,
        **kwargs,
    ) -> None:
        """Edit a chat message by ID using Markdown with plain-text fallback."""
        message_id_int = int(message_id)
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id_int,
                text=text,
                parse_mode=MARKDOWN_PARSE_MODE,
                **kwargs,
            )
        except BadRequest as exc:
            if "Can't parse entities" not in str(exc):
                raise
            logger.warning(
                "Markdown parse failed in edit_message_text by id, retrying plain text: %s",
                exc,
            )
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id_int,
                text=text,
                **kwargs,
            )

    @property
    def _orchestrator(self):
        """Return the bound Orchestrator instance if one is registered as handler, else None."""
        bound_self = getattr(self._handler, "__self__", None)
        if bound_self is not None and bound_self.__class__.__name__ == "Orchestrator":
            return bound_self
        return None

    async def _handle_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Catch-all handler for slash-commands delegated to orchestrator.

        Extracts the bare command name from the incoming message and delegates
        to ``Orchestrator.handle_admin_command()``.  Any client that integrates
        admin commands only needs to hook into that single orchestrator method.
        """
        if not update.message:
            return

        raw = (update.message.text or "").strip()
        # Handle both /cmd and /cmd@botname forms
        command = raw.lstrip("/").split("@")[0].split()[0].lower() if raw.startswith("/") else ""
        args = list(context.args or [])
        chat_id = str(update.message.chat.id)

        orch = self._orchestrator
        if orch is not None:
            response = await orch.handle_admin_command(chat_id, command, args)
        else:
            response = "Admin commands are only available in orchestrator mode."

        await self._reply_text_safe(update.message, response)

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
                    message_context={
                        "platform": "telegram",
                        "chat_type": getattr(update.message.chat, "type", None),
                        "effective_user_id": getattr(update.effective_user, "id", None),
                    },
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
                approval_message = await self._reply_text_safe(
                    update.message,
                    reply.content,
                    reply_markup=reply_markup,
                )
                if approval_message is not None and getattr(approval_message, "message_id", None) is not None:
                    self._approval_messages[reply.approval_id] = (
                        chat_id,
                        str(approval_message.message_id),
                    )
                if reply.images:
                    await self._send_images_with_bot(context.bot, chat_id, reply.images)
            else:
                # No approval needed, just send the content
                await self._reply_text_safe(
                    update.message,
                    reply.content,
                )
                if reply.images:
                    await self._send_images_with_bot(context.bot, chat_id, reply.images)
        else:
            # Plain string response (backward compatibility)
            await self._reply_text_safe(update.message, reply)

    @staticmethod
    def _photo_payload_from_url(image_url: str) -> str | InputFile:
        """Convert a URL/data URL into a Telegram send_photo payload."""
        if not image_url.startswith("data:"):
            return image_url

        header, encoded = image_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Unsupported non-base64 data URL")

        media_type = "image/jpeg"
        if header.startswith("data:"):
            media_type = header[5:].split(";", 1)[0] or media_type

        payload = base64.b64decode(encoded)
        extension = mimetypes.guess_extension(media_type) or ".jpg"
        return InputFile(payload, filename=f"reference{extension}")

    async def _send_images_with_bot(self, bot: Any, chat_id: str, image_urls: list[str]) -> list[dict[str, str]]:
        """Send images using a specific bot instance and collect metadata."""
        sent: list[dict[str, str]] = []
        for image_url in image_urls:
            try:
                payload = self._photo_payload_from_url(image_url)
                message = await bot.send_photo(chat_id=chat_id, photo=payload)
                sent.append({
                    "message_id": str(message.message_id),
                    "timestamp": dt.now(timezone.utc).isoformat(),
                })
            except Exception as exc:
                logger.warning("Failed to send reference image: %s", exc)
        return sent

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
                reply = await self._handler(
                    chat_id,
                    command_text,
                    message_context={
                        "platform": "telegram",
                        "chat_type": getattr(query.message.chat, "type", None) if query.message else None,
                        "effective_user_id": getattr(update.effective_user, "id", None),
                    },
                )
            else:
                reply = await self._handler(command_text)
        except Exception as exc:
            logger.exception("Error from LLM handler during callback: %s", exc)
            reply = "⚠️ Sorry, I encountered an error processing your decision."

        # Extract content if reply is MessageResponse
        reply_text = reply.content if isinstance(reply, MessageResponse) else reply

        self._approval_messages.pop(approval_id, None)

        # Keep final responses as normal messages after approval handling.
        if chat_id and reply_text:
            await self._send_message_safe(context.bot, chat_id, str(reply_text))

    async def notify_tool_status(
        self,
        user_id: str,
        status: str,
        tool_name: str,
        *,
        approval_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Publish interim tool lifecycle feedback to Telegram users."""
        if status == "running":
            text = f"🔨 Running: `{tool_name}`"
        elif status == "failed":
            text = f"❌ Failed: `{tool_name}`"
            if detail:
                text += f"\n{detail}"
        else:
            return

        bot = self._app.bot if self._app is not None else None

        if status == "running" and approval_id and approval_id in self._approval_messages and bot is not None:
            approval_chat_id, approval_message_id = self._approval_messages[approval_id]
            try:
                await self._edit_chat_message_text_safe(
                    bot,
                    approval_chat_id,
                    approval_message_id,
                    text,
                )
                return
            except Exception as exc:
                logger.warning("Could not edit approval message for running status: %s", exc)

        if bot is not None:
            await self._send_message_safe(bot, user_id, text)
            return

        await self.send_message(user_id, text)

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

        # Catch-all for slash commands (/start, /status, /clear, /tools, /reload, …).
        self._app.add_handler(MessageHandler(filters.COMMAND, self._handle_command))
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
                "timestamp": dt.now(timezone.utc).isoformat(),
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

    async def send_images(self, user_id: str, image_urls: list[str]) -> dict:
        """Send one or more images to a specific Telegram chat."""
        if self._app is None:
            self._app = Application.builder().token(self._config.bot_token).build()
            await self._app.initialize()

        sent_items = await self._send_images_with_bot(self._app.bot, user_id, image_urls)
        return {
            "status": "sent",
            "user_id": user_id,
            "count": len(sent_items),
            "items": sent_items,
        }

    async def send_file(self, user_id: str, file_path: str) -> dict:
        """Send a local file to a specific Telegram chat as a document."""
        resolved_path = Path(file_path).expanduser()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if self._app is None:
            self._app = Application.builder().token(self._config.bot_token).build()
            await self._app.initialize()

        try:
            with resolved_path.open("rb") as handle:
                payload = InputFile(handle, filename=resolved_path.name)
                message = await self._app.bot.send_document(
                    chat_id=user_id,
                    document=payload,
                )

            document = getattr(message, "document", None)
            file_id = getattr(document, "file_id", None)
            return {
                "message_id": str(message.message_id),
                "timestamp": dt.now(timezone.utc).isoformat(),
                "status": "sent",
                "user_id": user_id,
                "file_name": resolved_path.name,
                "file_id": str(file_id) if file_id else None,
            }
        except BadRequest as exc:
            logger.error(
                "Failed to send file to user %s: %s",
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
