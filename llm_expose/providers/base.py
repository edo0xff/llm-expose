"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

Message = dict[str, Any]
ToolSpec = dict[str, Any]
ToolChoice = str | dict[str, Any]


class BaseProvider(ABC):
    """Common interface that all LLM provider adapters must implement.

    Providers are responsible for sending a conversation history to an LLM
    and returning the model's reply. The conversation history follows the
    OpenAI-style message format::

        [{"role": "user", "content": "Hello!"}]
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> str:
        """Return a completion for *messages*.

        Args:
            messages: A list of message dicts following the OpenAI chat
                completion format.
            tools: Optional tool definitions supported by the provider.
            tool_choice: Optional tool selection policy.

        Returns:
            The model's reply as a plain string.
        """

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> AsyncIterator[str]:
        """Yield completion tokens for *messages* as they arrive.

        Args:
            messages: Same format as :meth:`complete`.
            tools: Optional tool definitions supported by the provider.
            tool_choice: Optional tool selection policy.

        Yields:
            Individual text tokens/chunks from the model response.
        """

    def supports_vision(self) -> bool:
        """Return whether the configured model supports image input."""
        return False
