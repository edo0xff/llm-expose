"""Abstract base class for messaging client adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Coroutine, Any

# Type alias: an async function that accepts a channel ID and user message, and returns a reply.
MessageHandler = Callable[[str, str], Coroutine[Any, Any, str]]


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
