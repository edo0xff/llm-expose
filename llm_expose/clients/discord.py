"""Discord client adapter using discord.py."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import mimetypes
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import discord

from llm_expose.clients.base import BaseClient, MessageResponse
from llm_expose.clients.base import MessageHandler as LLMHandler
from llm_expose.config.models import DiscordClientConfig
from llm_expose.core.content_parts import build_user_content

logger = logging.getLogger(__name__)

# Discord per-message character limit
_DISCORD_MAX_LEN = 2000
# Small delay between consecutive chunk sends to stay within rate limits
_CHUNK_DELAY = 0.5


def _chunk_text(text: str, max_len: int = _DISCORD_MAX_LEN) -> list[str]:
    """Split *text* into chunks that fit within Discord's message limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks


class _ApprovalView(discord.ui.View):
    """Discord UI View with Approve/Reject buttons for tool execution approval."""

    def __init__(
        self, client: DiscordClient, approval_id: str, timeout: float = 600.0
    ) -> None:
        super().__init__(timeout=timeout)
        self._client = client
        self._approval_id = approval_id

    async def _handle(self, interaction: discord.Interaction, decision: str) -> None:
        """Shared logic for both button callbacks."""
        # Disable all buttons immediately so they can't be pressed twice
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        await interaction.response.edit_message(view=self)
        self.stop()

        channel_id = str(interaction.channel_id)
        command_text = f"{decision} {self._approval_id}"
        logger.info(
            "Discord approval button pressed: %s in channel %s",
            command_text,
            channel_id,
        )

        try:
            bound_self = getattr(self._client._handler, "__self__", None)
            if (
                bound_self is not None
                and bound_self.__class__.__name__ == "Orchestrator"
            ):
                reply = await self._client._handler(
                    channel_id,
                    command_text,
                    message_context={
                        "platform": "discord",
                        "channel_id": channel_id,
                    },
                )
            else:
                reply = await self._client._handler(command_text)
        except Exception as exc:
            logger.exception("Error from handler during Discord approval: %s", exc)
            reply = "⚠️ Sorry, I encountered an error processing your decision."

        reply_text = reply.content if isinstance(reply, MessageResponse) else str(reply)
        self._client._approval_messages.pop(self._approval_id, None)

        if interaction.channel and reply_text:
            for chunk in _chunk_text(reply_text):
                await interaction.channel.send(chunk)  # type: ignore[union-attr]
                if len(_chunk_text(reply_text)) > 1:
                    await asyncio.sleep(_CHUNK_DELAY)

    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.success)
    async def approve_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:  # noqa: ARG002
        await self._handle(interaction, "approve")

    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.danger)
    async def reject_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:  # noqa: ARG002
        await self._handle(interaction, "reject")


class DiscordClient(BaseClient):
    """Messaging client adapter for Discord.

    Listens for incoming messages in guild channels and DMs via the Discord
    Gateway (WebSocket) and forwards them to the registered LLM handler.

    **Required Discord bot permissions:**
    - ``Read Messages / View Channels``
    - ``Send Messages``
    - ``Read Message History``
    - ``Add Reactions`` *(optional, for future use)*

    **Required Privileged Intent:**
    Enable *Message Content Intent* in the Discord Developer Portal for
    the bot to receive ``message.content``.

    Args:
        config: Discord-specific configuration (bot token).
        handler: Async callable that receives the user's text and returns the
            LLM's reply.
    """

    def __init__(self, config: DiscordClientConfig, handler: LLMHandler) -> None:
        super().__init__(handler)
        self._config = config
        self._bot: discord.Client | None = None
        self._stop_event: asyncio.Event | None = None
        # Maps approval_id → (channel_id, message_id) for notify_tool_status edits
        self._approval_messages: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _orchestrator(self) -> Any | None:
        """Return the bound Orchestrator if one is registered as handler, else None."""
        bound_self = getattr(self._handler, "__self__", None)
        if bound_self is not None and bound_self.__class__.__name__ == "Orchestrator":
            return bound_self
        return None

    @staticmethod
    def _build_intents() -> discord.Intents:
        """Build explicit intents for guild/DM message handling.

        We avoid relying on discord.py defaults so message event subscription is
        stable across library versions.
        """
        intents = discord.Intents.default()
        intents.message_content = (
            True  # Privileged intent; must also be enabled in Portal
        )

        # Make message event subscriptions explicit when attributes exist.
        for attr in ("messages", "guild_messages", "dm_messages"):
            if hasattr(intents, attr):
                setattr(intents, attr, True)

        return intents

    async def _get_channel(self, channel_id: str) -> discord.abc.Messageable | None:
        """Fetch a Discord channel/DM by ID, using cache first."""
        if self._bot is None:
            return None
        channel_int = int(channel_id)
        channel = self._bot.get_channel(channel_int)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_int)
            except Exception as exc:
                logger.warning(
                    "Could not fetch Discord channel %s: %s", channel_id, exc
                )
                return None
        return channel  # type: ignore[return-value]

    async def _send_text_to_channel(
        self,
        channel: discord.abc.Messageable,
        text: str,
        *,
        view: discord.ui.View | None = None,
    ) -> discord.Message | None:
        """Send text to a Discord channel, chunking if needed.

        Returns the *last* Message sent (useful for storing the message_id).
        """
        chunks = _chunk_text(text)
        last_message: discord.Message | None = None
        for i, chunk in enumerate(chunks):
            kwargs: dict[str, Any] = {}
            # Only attach the View to the first chunk (where buttons make sense)
            if view is not None and i == 0:
                kwargs["view"] = view
            last_message = await channel.send(chunk, **kwargs)
            if i < len(chunks) - 1:
                await asyncio.sleep(_CHUNK_DELAY)
        return last_message

    # ------------------------------------------------------------------
    # Discord event handlers (registered during start)
    # ------------------------------------------------------------------

    async def _on_ready(self) -> None:
        if self._bot is None:
            logger.info("Discord on_ready received but bot is not initialized")
            return

        logger.info(
            "Discord bot connected as %s (id=%s), joined %d guild(s)",
            self._bot.user,
            self._bot.user.id if self._bot and self._bot.user else "?",
            len(self._bot.guilds) if self._bot else 0,
        )

    async def _on_message(self, message: discord.Message) -> None:
        """Handle incoming messages from guild channels and DMs."""
        # Ignore messages from bots (includes self)
        if message.author.bot:
            return

        channel_id = str(message.channel.id)
        user_text = message.content or ""

        # Extract image attachments as data URLs (jpg/png/gif/webp)
        image_urls = await self._extract_image_data_urls(message)

        if not user_text and not image_urls:
            logger.info(
                "Skipping Discord message with no text and no image attachments "
                "(channel=%s, guild=%s)",
                channel_id,
                "dm" if message.guild is None else str(message.guild.id),
            )
            return

        logger.info(
            "Discord message from %s in channel %s",
            message.author,
            channel_id,
        )

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                if self._orchestrator is not None:
                    message_content = build_user_content(
                        user_text, image_urls=image_urls
                    )
                    reply = await self._handler(
                        channel_id,
                        user_text,
                        message_content=message_content,
                        message_context={
                            "platform": "discord",
                            "channel_id": channel_id,
                            "guild_id": (
                                str(message.guild.id) if message.guild else None
                            ),
                        },
                    )
                else:
                    reply = await self._handler(user_text)
            except Exception as exc:
                logger.exception("Error from LLM handler: %s", exc)
                reply = "⚠️ Sorry, I encountered an error. Please try again."

        # Handle structured MessageResponse (approval workflow)
        if isinstance(reply, MessageResponse):
            if reply.approval_id:
                view = _ApprovalView(self, reply.approval_id)
                sent = await self._send_text_to_channel(
                    message.channel,
                    reply.content,
                    view=view,
                )
                if sent is not None:
                    self._approval_messages[reply.approval_id] = (
                        channel_id,
                        str(sent.id),
                    )
                if reply.images:
                    await self._send_images_to_channel(
                        message.channel,
                        reply.images,
                    )
            else:
                await self._send_text_to_channel(
                    message.channel,
                    reply.content,
                )
                if reply.images:
                    await self._send_images_to_channel(
                        message.channel,
                        reply.images,
                    )
        else:
            await self._send_text_to_channel(
                message.channel,
                str(reply),
            )

    @staticmethod
    async def _extract_image_data_urls(message: discord.Message) -> list[str]:
        """Download image attachments and return them as data URLs."""
        image_urls: list[str] = []
        for attachment in message.attachments:
            content_type = attachment.content_type or ""
            if not content_type.startswith("image/"):
                continue
            try:
                payload = await attachment.read()
                encoded = base64.b64encode(payload).decode("ascii")
                mime = content_type.split(";", 1)[0].strip() or "image/jpeg"
                image_urls.append(f"data:{mime};base64,{encoded}")
            except Exception as exc:
                logger.warning(
                    "Failed to read Discord attachment %s: %s", attachment.filename, exc
                )
        return image_urls

    async def _send_images_to_channel(
        self, channel: discord.abc.Messageable, image_urls: list[str]
    ) -> None:
        """Send images to a Discord channel as file attachments."""
        for image_url in image_urls:
            try:
                if image_url.startswith("data:"):
                    header, encoded = image_url.split(",", 1)
                    payload = base64.b64decode(encoded)
                    mime = "image/jpeg"
                    if header.startswith("data:"):
                        mime = header[5:].split(";", 1)[0] or mime
                    ext = mimetypes.guess_extension(mime) or ".jpg"
                    fp = io.BytesIO(payload)
                    await channel.send(file=discord.File(fp, filename=f"image{ext}"))
                else:
                    # Remote URL — just embed as a link so Discord previews it
                    await channel.send(image_url)
            except Exception as exc:
                logger.warning("Failed to send image to Discord: %s", exc)

    # ------------------------------------------------------------------
    # notify_tool_status (BaseClient interface)
    # ------------------------------------------------------------------

    async def notify_tool_status(
        self,
        user_id: str,
        status: str,
        tool_name: str,
        *,
        approval_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Publish interim tool lifecycle feedback to Discord users."""
        if status == "running":
            text = f"🔨 Running: `{tool_name}`"
        elif status == "failed":
            text = f"❌ Failed: `{tool_name}`"
            if detail:
                text += f"\n{detail}"
        else:
            return

        # Try editing the approval message first (consistent with Telegram)
        if (
            status == "running"
            and approval_id
            and approval_id in self._approval_messages
            and self._bot is not None
        ):
            approval_channel_id, approval_message_id = self._approval_messages[
                approval_id
            ]
            try:
                channel = await self._get_channel(approval_channel_id)
                if channel is not None:
                    msg = await channel.fetch_message(int(approval_message_id))
                    await msg.edit(content=text)
                    return
            except Exception as exc:
                logger.warning(
                    "Could not edit approval message for running status: %s", exc
                )

        # Fall back to sending a new message
        channel = await self._get_channel(user_id)
        if channel is not None:
            await channel.send(text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Discord Gateway and start receiving messages."""
        self._stop_event = asyncio.Event()

        intents = self._build_intents()
        logger.info(
            "Discord intents configured: message_content=%s, messages=%s, "
            "guild_messages=%s, dm_messages=%s",
            getattr(intents, "message_content", None),
            getattr(intents, "messages", None),
            getattr(intents, "guild_messages", None),
            getattr(intents, "dm_messages", None),
        )

        self._bot = discord.Client(intents=intents)

        # Register event handlers using decorators.
        # discord.Client doesn't have add_listener(); we use @event decorators instead.
        @self._bot.event
        async def on_ready() -> None:
            await self._on_ready()

        @self._bot.event
        async def on_message(message: discord.Message) -> None:
            await self._on_message(message)

        logger.info("Starting Discord bot…")
        try:
            await self._bot.start(self._config.bot_token)
        finally:
            self._stop_event.set()

    async def stop(self) -> None:
        """Disconnect from Discord Gateway and release resources."""
        if self._bot is None:
            return
        logger.info("Stopping Discord bot…")
        await self._bot.close()
        self._bot = None
        if self._stop_event is not None:
            self._stop_event.set()
            self._stop_event = None

    # ------------------------------------------------------------------
    # Direct send methods (BaseClient interface)
    # ------------------------------------------------------------------

    async def _ensure_bot_ready(self) -> discord.Client:
        """Return a ready bot instance, initializing one if needed for one-shot sends."""
        if self._bot is not None:
            return self._bot

        intents = self._build_intents()
        bot = discord.Client(intents=intents)

        # Login only (no gateway connection) is not directly supported by discord.py's
        # high-level Client in the same way as Telegram's send-only flow.
        # We connect and immediately use the bot, relying on the caller to manage the
        # event loop lifetime (same pattern as Telegram's lazy init).
        self._bot = bot
        await bot.login(self._config.bot_token)
        return bot

    async def send_message(self, user_id: str, text: str) -> dict:
        """Send a direct message to a specific Discord channel/DM.

        Args:
            user_id: Discord channel ID (as string) to send to.
            text: Message text to send.

        Returns:
            Dict with keys: message_id, timestamp, status, user_id.
        """
        bot = await self._ensure_bot_ready()
        channel = bot.get_channel(int(user_id))
        if channel is None:
            channel = await bot.fetch_channel(int(user_id))

        last_msg = await self._send_text_to_channel(channel, text)  # type: ignore[arg-type]
        message_id = str(last_msg.id) if last_msg else "unknown"
        return {
            "message_id": message_id,
            "timestamp": dt.now(UTC).isoformat(),
            "status": "sent",
            "user_id": user_id,
        }

    async def send_images(self, user_id: str, image_urls: list[str]) -> dict:
        """Send one or more images to a specific Discord channel.

        Args:
            user_id: Discord channel ID as string.
            image_urls: Image references as remote URLs or data URLs.
        """
        bot = await self._ensure_bot_ready()
        channel = bot.get_channel(int(user_id))
        if channel is None:
            channel = await bot.fetch_channel(int(user_id))

        await self._send_images_to_channel(channel, image_urls)  # type: ignore[arg-type]
        return {
            "status": "sent",
            "user_id": user_id,
            "count": len(image_urls),
        }

    async def send_file(self, user_id: str, file_path: str) -> dict:
        """Send a local file to a specific Discord channel as an attachment.

        Args:
            user_id: Discord channel ID as string.
            file_path: Path to a local file.
        """
        resolved = Path(file_path).expanduser()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        bot = await self._ensure_bot_ready()
        channel = bot.get_channel(int(user_id))
        if channel is None:
            channel = await bot.fetch_channel(int(user_id))

        with resolved.open("rb") as handle:
            msg = await channel.send(file=discord.File(handle, filename=resolved.name))  # type: ignore[union-attr]

        return {
            "message_id": str(msg.id),
            "timestamp": dt.now(UTC).isoformat(),
            "status": "sent",
            "user_id": user_id,
            "file_name": resolved.name,
        }
