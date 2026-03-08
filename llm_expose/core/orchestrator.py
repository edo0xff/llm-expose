"""Orchestrator: ties together an LLM provider and a messaging client."""

from __future__ import annotations

import logging

from llm_expose.clients.base import BaseClient
from llm_expose.config.models import ExposureConfig
from llm_expose.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# System prompt injected at the start of every conversation.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
)


class Orchestrator:
    """Connects an LLM *provider* to a messaging *client*.

    The orchestrator maintains per-channel conversation history so that the
    LLM retains context across multiple turns in a single chat session. Each
    channel can have its own system prompt to provide context about the
    communication channel being used.

    Args:
        config: The :class:`~llm_expose.config.models.ExposureConfig` that
            describes both the provider and the client to use.
        provider: An initialised :class:`~llm_expose.providers.base.BaseProvider`.
        client: An initialised :class:`~llm_expose.clients.base.BaseClient`.
    """

    def __init__(
        self,
        config: ExposureConfig,
        provider: BaseProvider,
        client: BaseClient,
    ) -> None:
        self._config = config
        self._provider = provider
        self._client = client
        # Per-channel conversation histories (keyed by channel ID).
        # Each channel gets its own history with the configured system prompt.
        self._histories: dict[str, list[dict[str, str]]] = {}
        # Use custom system prompt from config, or fall back to default
        self._system_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        # Backward compatibility for older tests/callers that access _history
        # or call _handle_message(user_message) without a channel ID.
        self._default_channel_id = "__default__"
        self._history = self._get_or_create_history(self._default_channel_id)

    def _get_or_create_history(self, channel_id: str) -> list[dict[str, str]]:
        """Get the conversation history for a channel, creating it if needed.

        Args:
            channel_id: Unique identifier for the channel/chat.

        Returns:
            The conversation history list for this channel.
        """
        if channel_id not in self._histories:
            logger.debug("Creating new conversation history for channel %s", channel_id)
            self._histories[channel_id] = [
                {"role": "system", "content": self._system_prompt}
            ]
        return self._histories[channel_id]

    async def _handle_message(
        self, channel_or_message: str, user_message: str | None = None
    ) -> str:
        """Process a single user message and return the LLM's reply.

        Appends the user message to the channel's history, calls the provider,
        then appends the assistant's reply and returns it.

        Args:
            channel_or_message: Either channel ID (new style) or user message
                (legacy one-argument style).
            user_message: Raw text from the end user when channel ID is passed.

        Returns:
            The LLM's reply as a plain string.
        """
        if user_message is None:
            channel_id = self._default_channel_id
            text = channel_or_message
        else:
            channel_id = channel_or_message
            text = user_message

        history = self._get_or_create_history(channel_id)
        history.append({"role": "user", "content": text})
        logger.debug(
            "Sending %d messages to provider for channel %s",
            len(history),
            channel_id,
        )
        reply = await self._provider.complete(history)
        history.append({"role": "assistant", "content": reply})
        return reply

    async def run(self) -> None:
        """Start the client and block until it stops.

        Registers :meth:`_handle_message` as the client's message handler and
        delegates lifecycle management to the client.
        """
        # Wire our handler into the client via the public setter
        self._client.set_handler(self._handle_message)
        logger.info("Starting orchestrator for exposure '%s'", self._config.name)
        try:
            await self._client.start()
        finally:
            # Always attempt a graceful shutdown to release client resources
            # (e.g., Telegram long-polling session) even on Ctrl+C/errors.
            await self._client.stop()
