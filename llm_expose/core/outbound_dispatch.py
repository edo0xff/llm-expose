"""Shared outbound message delivery helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_expose.config.loader import get_pairs_for_channel, load_channel
from llm_expose.config.models import TelegramClientConfig

if TYPE_CHECKING:
    from llm_expose.clients.base import BaseClient


class OutboundMessageError(Exception):
    """Raised when an outbound message cannot be delivered."""


class OutboundMessagePermissionError(OutboundMessageError):
    """Raised when the target user is not authorized for the selected channel."""


async def _noop_handler(*args: Any, **kwargs: Any) -> str:
    del args, kwargs
    return ""


def _create_client(config: TelegramClientConfig) -> BaseClient:
    """Create a messaging client instance for the provided channel config."""
    if config.client_type == "telegram":
        from llm_expose.clients.telegram import TelegramClient

        return TelegramClient(config, handler=_noop_handler)
    raise OutboundMessageError(f"Unsupported client type '{config.client_type}'.")


async def dispatch_channel_message(
    channel_name: str,
    target_user_id: str,
    text: str,
    *,
    file_path: str | None = None,
    image_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Send a message to a paired recipient through a saved channel config."""
    normalized_channel_name = channel_name.strip()
    normalized_target_user_id = target_user_id.strip()

    if not normalized_channel_name:
        raise OutboundMessageError("Channel name cannot be empty.")
    if not normalized_target_user_id:
        raise OutboundMessageError("Target user ID cannot be empty.")
    if not text or not text.strip():
        raise OutboundMessageError("Message text cannot be empty.")

    try:
        client_config = load_channel(normalized_channel_name)
    except FileNotFoundError as exc:
        raise OutboundMessageError(
            f"Channel '{normalized_channel_name}' not found."
        ) from exc
    except Exception as exc:
        raise OutboundMessageError(
            f"Failed to load channel '{normalized_channel_name}': {exc}"
        ) from exc

    try:
        paired_users = get_pairs_for_channel(normalized_channel_name)
    except Exception as exc:
        raise OutboundMessageError(
            f"Failed to load pairing configuration for channel '{normalized_channel_name}': {exc}"
        ) from exc

    if normalized_target_user_id not in paired_users:
        raise OutboundMessagePermissionError(
            f"User '{normalized_target_user_id}' is not paired with channel '{normalized_channel_name}'."
        )

    client = _create_client(client_config)
    send_result = await client.send_message(normalized_target_user_id, text)

    if file_path is not None:
        file_result = await client.send_file(normalized_target_user_id, file_path)
        send_result["file_reference"] = file_result

    if image_urls:
        image_result = await client.send_images(normalized_target_user_id, image_urls)
        send_result["image_reference"] = image_result

    return send_result