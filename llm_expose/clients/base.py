"""Abstract base class for messaging client adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Coroutine, Any

@dataclass
class MessageResponse:
    """Structured response that can include approval metadata for interactive UI.
    
    Attributes:
        content: The text message to display to the user.
        approval_id: Optional approval ID for tool execution confirmation.
        tool_names: Optional list of tool names requiring approval.
        server_names: Optional mapping of tool names to server names.
    """
    content: str
    approval_id: str | None = None
    tool_names: list[str] | None = None
    server_names: dict[str, str] | None = None


# Backward-compatible handler signature support:
# - Legacy: handler(user_message) -> str
# - New: handler(channel_id, user_message) -> str | MessageResponse
MessageHandler = Callable[..., Coroutine[Any, Any, str | MessageResponse]]


class BaseClient(ABC):
    """Common interface that all messaging client adapters must implement.

    A client is responsible for:

    1. Connecting to the external messaging platform.
    2. Receiving user messages.
    3. Delegating them to the registered :data:`MessageHandler`.
    4. Sending the handler's reply back to the user.
    """

    def __init__(self, handler: MessageHandler) -> None:
        """Initialise the client with an async *handler*.

        Args:
            handler: Async callable that receives a user message string and
                returns the model's reply string.
        """
        self._handler = handler

    @abstractmethod
    async def start(self) -> None:
        """Connect to the platform and begin receiving messages.

        This method should block (run the event loop) until :meth:`stop` is
        called or an unrecoverable error occurs.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully disconnect from the platform and release resources."""

    def set_handler(self, handler: MessageHandler) -> None:
        """Replace the current message handler with *handler*.

        Args:
            handler: Async callable that receives a user message string and
                returns the model's reply string.
        """
        self._handler = handler
