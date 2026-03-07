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

    The orchestrator maintains per-session conversation history so that the
    LLM retains context across multiple turns in a single chat session.

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
        # Simple in-memory conversation history (resets on restart).
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}
        ]

    async def _handle_message(self, user_message: str) -> str:
        """Process a single user message and return the LLM's reply.

        Appends the user message to the history, calls the provider, then
        appends the assistant's reply and returns it.

        Args:
            user_message: Raw text from the end user.

        Returns:
            The LLM's reply as a plain string.
        """
        self._history.append({"role": "user", "content": user_message})
        logger.debug("Sending %d messages to provider", len(self._history))
        reply = await self._provider.complete(self._history)
        self._history.append({"role": "assistant", "content": reply})
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
