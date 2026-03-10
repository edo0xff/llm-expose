"""Utilities for OpenAI-compatible multimodal message content parts."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

Message = dict[str, Any]


def content_has_images(content: Any) -> bool:
    """Return True when content contains at least one image block."""
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                return True
    return False


def messages_have_images(messages: list[Message]) -> bool:
    """Return True when any message includes image blocks."""
    for message in messages:
        if content_has_images(message.get("content")):
            return True
    return False


def strip_image_parts(messages: list[Message]) -> tuple[list[Message], int]:
    """Return a copy of messages with all image parts removed.

    Returns:
        A tuple of (new_messages, stripped_count).
    """
    stripped_count = 0
    new_messages: list[Message] = []

    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            kept_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                        stripped_count += 1
                        continue
                kept_parts.append(part)

            if kept_parts:
                copied["content"] = kept_parts
            else:
                copied["content"] = ""
        new_messages.append(copied)

    return new_messages, stripped_count


def build_user_content(
    text: str | None,
    *,
    image_urls: list[str] | None = None,
    image_detail: str = "auto",
) -> str | list[dict[str, Any]]:
    """Build user content string or multimodal content list."""
    normalized_text = (text or "").strip()
    urls = [url for url in (image_urls or []) if url and url.strip()]

    if not urls:
        return normalized_text

    parts: list[dict[str, Any]] = []
    if normalized_text:
        parts.append({"type": "text", "text": normalized_text})

    for url in urls:
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": image_detail,
                },
            }
        )

    return parts


def file_to_data_url(path: str | Path) -> str:
    """Convert a local file into a data URL for image_url blocks."""
    file_path = Path(path)
    data = file_path.read_bytes()
    media_type, _ = mimetypes.guess_type(str(file_path))
    if not media_type:
        media_type = "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def normalize_mcp_content(content: Any) -> list[dict[str, Any]]:
    """Normalize MCP content blocks into OpenAI-compatible text/image_url parts."""
    if not isinstance(content, list):
        return []

    normalized: list[dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            text_value = str(item)
            if text_value:
                normalized.append({"type": "text", "text": text_value})
            continue

        item_type = item.get("type")

        if item_type == "text":
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value:
                normalized.append({"type": "text", "text": text_value})
            continue

        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                normalized.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url["url"],
                            "detail": image_url.get("detail", "auto"),
                        },
                    }
                )
            continue

        if item_type == "image":
            source = item.get("source")
            if not isinstance(source, dict):
                continue

            source_type = source.get("type")
            if source_type == "url":
                url = source.get("url")
                if isinstance(url, str) and url:
                    normalized.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url, "detail": "auto"},
                        }
                    )
                continue

            # MCP image source commonly uses inline base64 payloads.
            if source_type in {"base64", "data"}:
                data = source.get("data")
                if not isinstance(data, str) or not data:
                    continue
                media_type = source.get("media_type") or source.get("mime_type") or "image/png"
                normalized.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}",
                            "detail": "auto",
                        },
                    }
                )

    return normalized
