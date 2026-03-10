"""Builtin in-process MCP tools for llm-expose."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from llm_expose.config.loader import get_pairs_for_channel
from llm_expose.core.content_parts import file_to_data_url


class _MessageSenderProtocol:
    async def send_message(self, user_id: str, text: str) -> dict[str, Any]:
        raise NotImplementedError

    async def send_file(self, user_id: str, file_path: str) -> dict[str, Any]:
        raise NotImplementedError

    async def send_images(self, user_id: str, image_urls: list[str]) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class ToolExecutionContext:
    """Execution metadata passed to builtin MCP tools."""

    execution_mode: Literal["chat", "one-shot"]
    channel_id: str
    channel_name: str | None = None
    subject_id: str | None = None
    subject_kind: Literal["user", "group", "chat", "unknown"] = "unknown"
    initiator_user_id: str | None = None
    platform: str | None = None
    chat_type: str | None = None
    sender: _MessageSenderProtocol | None = None
    invoked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable public view of the execution context."""
        user_id = self.subject_id if self.subject_kind == "user" else None
        group_id = self.subject_id if self.subject_kind == "group" else None
        return {
            "execution_mode": self.execution_mode,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "user_id": user_id,
            "group_id": group_id,
            "initiator_user_id": self.initiator_user_id,
            "platform": self.platform,
            "chat_type": self.chat_type,
            "invoked_at": self.invoked_at,
        }


@dataclass(slots=True)
class _BuiltinTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class _GetInvocationContextTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_get_invocation_context",
            description=(
                "Return llm-expose invocation context for the current conversation, "
                "including channel identifiers, execution mode, and UTC timestamp."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        del arguments
        payload = execution_context.to_public_dict() if execution_context is not None else {
            "execution_mode": "unknown",
            "channel_id": None,
            "channel_name": None,
            "subject_id": None,
            "subject_kind": "unknown",
            "user_id": None,
            "group_id": None,
            "initiator_user_id": None,
            "platform": None,
            "chat_type": None,
            "invoked_at": datetime.now(UTC).isoformat(),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, sort_keys=True),
                }
            ]
        }


class _GetPairingIdsTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_get_pairing_ids",
            description=(
                "Return the pairing IDs configured for the current channel. "
                "Pairing IDs represent the allowed sender/channel identifiers "
                "that are permitted to interact with this channel."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        del arguments
        if execution_context is None:
            return _error_result("execution context unavailable in this execution mode")
        channel_name = execution_context.channel_name
        if not channel_name:
            return _error_result("channel_name is not available in the current execution context")
        try:
            pairing_ids = get_pairs_for_channel(channel_name)
        except Exception as exc:
            return _error_result(f"failed to load pairing IDs: {exc}")
        payload = {
            "channel_name": channel_name,
            "pairing_ids": pairing_ids,
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, sort_keys=True),
                }
            ]
        }


class _SendTextMessageTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_send_text_message",
            description="Send a plain text message to a specific channel/user ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Target channel identifier for traceability.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Target user/chat identifier that receives the text.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Plain text message body to send.",
                    },
                },
                "required": ["channel_id", "user_id", "text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        sender = execution_context.sender if execution_context is not None else None
        if sender is None:
            return _error_result("builtin sender unavailable in this execution mode")

        channel_id = _required_string(arguments, "channel_id")
        user_id = _required_string(arguments, "user_id")
        text = _required_string(arguments, "text")
        if not channel_id or not user_id or not text:
            return _error_result("channel_id, user_id, and text are required")

        send_result = await sender.send_message(user_id, text)
        return _success_result(
            self.name,
            channel_id,
            user_id,
            send_result,
        )


class _SendFileMessageTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_send_file_message",
            description="Send a local file from path to a specific channel/user ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Target channel identifier for traceability.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Target user/chat identifier that receives the file.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or user-home-relative path to a local file.",
                    },
                },
                "required": ["channel_id", "user_id", "file_path"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        sender = execution_context.sender if execution_context is not None else None
        if sender is None:
            return _error_result("builtin sender unavailable in this execution mode")

        channel_id = _required_string(arguments, "channel_id")
        user_id = _required_string(arguments, "user_id")
        file_path_value = _required_string(arguments, "file_path")
        if not channel_id or not user_id or not file_path_value:
            return _error_result("channel_id, user_id, and file_path are required")

        file_path = Path(file_path_value).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            return _error_result(f"file not found: {file_path}")

        send_result = await sender.send_file(user_id, str(file_path))
        return _success_result(
            self.name,
            channel_id,
            user_id,
            send_result,
        )


class _SendImageMessageTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_send_image_message",
            description="Send an image from local path to a specific channel/user ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Target channel identifier for traceability.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Target user/chat identifier that receives the image.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Absolute or user-home-relative path to a local image file.",
                    },
                },
                "required": ["channel_id", "user_id", "image_path"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        sender = execution_context.sender if execution_context is not None else None
        if sender is None:
            return _error_result("builtin sender unavailable in this execution mode")

        channel_id = _required_string(arguments, "channel_id")
        user_id = _required_string(arguments, "user_id")
        image_path_value = _required_string(arguments, "image_path")
        if not channel_id or not user_id or not image_path_value:
            return _error_result("channel_id, user_id, and image_path are required")

        image_path = Path(image_path_value).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            return _error_result(f"file not found: {image_path}")

        data_url = file_to_data_url(image_path)
        send_result = await sender.send_images(user_id, [data_url])
        return _success_result(
            self.name,
            channel_id,
            user_id,
            send_result,
        )


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _error_result(error: str) -> dict[str, Any]:
    payload = {"status": "error", "error": error}
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, sort_keys=True),
            }
        ]
    }


def _success_result(
    tool_name: str,
    channel_id: str,
    user_id: str,
    send_result: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "tool": tool_name,
        "channel_id": channel_id,
        "user_id": user_id,
        "result": send_result,
    }
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, sort_keys=True),
            }
        ]
    }


class BuiltinMCPClient:
    """Minimal in-process MCP-like client for builtin llm-expose tools."""

    def __init__(self, server_name: str) -> None:
        self._server_name = server_name
        self._tools: dict[str, _BuiltinTool] = {
            tool.name: tool
            for tool in [
                _GetInvocationContextTool(),
                _GetPairingIdsTool(),
                _SendTextMessageTool(),
                _SendFileMessageTool(),
                _SendImageMessageTool(),
            ]
        }

    async def __aenter__(self) -> "BuiltinMCPClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def list_prompts(self) -> list[dict[str, Any]]:
        return []

    async def get_prompt(self, prompt_name: str) -> dict[str, Any]:
        raise ValueError(f"Builtin MCP server '{self._server_name}' has no prompt named '{prompt_name}'.")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.call_tool_with_context(
            tool_name,
            arguments,
            execution_context=None,
        )

    async def call_tool_with_context(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown builtin MCP tool '{tool_name}'.")
        return await tool.execute(arguments, execution_context=execution_context)