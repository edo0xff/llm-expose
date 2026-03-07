"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseProvider(ABC):
    """Common interface that all LLM provider adapters must implement.

    Providers are responsible for sending a conversation history to an LLM
    and returning the model's reply. The conversation history follows the
    OpenAI-style message format::

        [{"role": "user", "content": "Hello!"}]
    """

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a completion for *messages*.

        Args:
            messages: A list of message dicts with ``role`` and ``content``
                keys following the OpenAI chat completion format.

        Returns:
            The model's reply as a plain string.
        """

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield completion tokens for *messages* as they arrive.

        Args:
            messages: Same format as :meth:`complete`.

        Yields:
            Individual text tokens/chunks from the model response.
        """
