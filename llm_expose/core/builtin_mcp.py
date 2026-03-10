"""Builtin in-process MCP tools for llm-expose."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from llm_expose.core.outbound_dispatch import (
    OutboundMessageError,
    dispatch_channel_message,
)


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


def _json_text_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON payload in MCP text content blocks."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, sort_keys=True),
            }
        ]
    }


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
        return _json_text_result(payload)


class _SendMessageTool(_BuiltinTool):
    def __init__(self) -> None:
        super().__init__(
            name="llm_expose_send_message",
            description=(
                "Send a text message to a paired recipient in the current channel. "
                "The recipient must already be paired with that channel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target_user_id": {
                        "type": "string",
                        "description": "Recipient user/chat ID already paired with the current channel.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to send.",
                    },
                },
                "required": ["target_user_id", "text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        execution_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        channel_name = execution_context.channel_name if execution_context is not None else None
        target_user_id = str(arguments.get("target_user_id", "")).strip()
        text = str(arguments.get("text", ""))

        if not channel_name:
            return _json_text_result(
                {
                    "status": "error",
                    "error": "Cannot send a message because the current channel context is unavailable.",
                    "target_user_id": target_user_id or None,
                }
            )

        if not target_user_id:
            return _json_text_result(
                {
                    "status": "error",
                    "error": "target_user_id is required.",
                    "channel_name": channel_name,
                }
            )

        if not text.strip():
            return _json_text_result(
                {
                    "status": "error",
                    "error": "text is required.",
                    "channel_name": channel_name,
                    "target_user_id": target_user_id,
                }
            )

        try:
            payload = await dispatch_channel_message(
                channel_name,
                target_user_id,
                text,
            )
        except OutboundMessageError as exc:
            payload = {
                "status": "error",
                "error": str(exc),
                "channel_name": channel_name,
                "target_user_id": target_user_id,
            }

        return _json_text_result(payload)


class BuiltinMCPClient:
    """Minimal in-process MCP-like client for builtin llm-expose tools."""

    def __init__(self, server_name: str) -> None:
        self._server_name = server_name
        self._tools: dict[str, _BuiltinTool] = {
            tool.name: tool
            for tool in [
                _GetInvocationContextTool(),
                _SendMessageTool(),
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