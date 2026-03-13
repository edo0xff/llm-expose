"""Abstract base class for messaging client adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass
class MessageResponse:
    """Structured response that can include approval metadata for interactive UI.

    Attributes:
        content: The text message to display to the user.
        images: Optional list of image URLs/data URLs to send as references.
        approval_id: Optional approval ID for tool execution confirmation.
        tool_names: Optional list of tool names requiring approval.
        server_names: Optional mapping of tool names to server names.
    """

    content: str
    images: list[str] | None = None
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

    @abstractmethod
    async def send_message(self, user_id: str, text: str) -> dict:
        """Send a direct message to a specific user.

        This method sends a plain text message directly to a user, bypassing
        the regular message handler. It is intended for programmatic message
        delivery (e.g., cron jobs, scheduled notifications).

        Args:
            user_id: The user/chat ID to send the message to.
            text: The message text to send. May include Markdown formatting
                if the client supports it.

        Returns:
            A dict with keys:
                - message_id: The platform-specific message ID (str)
                - timestamp: ISO8601 timestamp when message was sent (str)
                - status: Status message (str, e.g. \"sent\")
                - user_id: Echo of the user_id parameter (str)

        Raises:
            Exception: Platform-specific exceptions on send failures (network,
                authentication, rate limits, invalid user_id).
        """

    @abstractmethod
    async def send_images(self, user_id: str, image_urls: list[str]) -> dict:
        """Send one or more images directly to a specific user.

        Args:
            user_id: The user/chat ID to send images to.
            image_urls: Image references as remote URLs or data URLs.

        Returns:
            A dict with provider-specific send metadata.
        """

    @abstractmethod
    async def send_file(self, user_id: str, file_path: str) -> dict:
        """Send a local file directly to a specific user.

        Args:
            user_id: The user/chat ID to send the file to.
            file_path: Path to a local file.

        Returns:
            A dict with provider-specific send metadata.
        """

    async def notify_tool_status(
        self,
        user_id: str,
        status: str,
        tool_name: str,
        *,
        approval_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Publish tool lifecycle status feedback to end users.

        Clients may choose how to present updates (new messages, message edits,
        inline UI updates, etc.). The default implementation is a no-op to keep
        backward compatibility for clients that do not support interim feedback.

        Args:
            user_id: Platform-specific chat/user identifier.
            status: Lifecycle state (for example: "running", "failed").
            tool_name: Name of the tool being executed.
            approval_id: Optional approval identifier when this status relates
                to an approval workflow.
            detail: Optional human-readable extra detail for the status.
        """
        return None

    def set_handler(self, handler: MessageHandler) -> None:
        """Replace the current message handler with *handler*.

        Args:
            handler: Async callable that receives a user message string and
                returns the model's reply string.
        """
        self._handler = handler
