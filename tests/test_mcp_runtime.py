"""Tests for builtin MCP runtime behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from llm_expose.config.models import MCPConfig, MCPServerConfig
from llm_expose.core.builtin_mcp import ToolExecutionContext
from llm_expose.core.outbound_dispatch import OutboundMessagePermissionError
from llm_expose.core.mcp_runtime import MCPRuntimeManager


class TestBuiltinMCPRuntime:
    @pytest.mark.asyncio
    async def test_builtin_server_registers_context_tool(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(name="builtin-core", transport="builtin", enabled=True),
                ]
            )
        )

        await runtime.initialize()

        assert runtime.tools == [
            {
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_context",
                    "description": (
                        "Return llm-expose invocation context for the current conversation, "
                        "including channel identifiers, execution mode, and UTC timestamp."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "llm_expose_send_message",
                    "description": (
                        "Send a text message to a paired recipient in the current channel. "
                        "The recipient must already be paired with that channel."
                    ),
                    "parameters": {
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
                },
            }
        ]

        result = await runtime.execute_tool_call(
            {
                "id": "call_ctx",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_context",
                    "arguments": "{}",
                },
            },
            execution_context=ToolExecutionContext(
                execution_mode="chat",
                channel_id="42",
                channel_name="telegram-main",
                subject_id="42",
                subject_kind="user",
                initiator_user_id="42",
                platform="telegram",
                chat_type="private",
            ),
        )

        payload = json.loads(result)
        assert payload["channel_id"] == "42"
        assert payload["channel_name"] == "telegram-main"
        assert payload["user_id"] == "42"
        assert payload["execution_mode"] == "chat"
        assert payload["platform"] == "telegram"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_server_respects_allowed_tools(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core",
                        transport="builtin",
                        enabled=True,
                        allowed_tools=["some_other_tool"],
                    ),
                ]
            )
        )

        await runtime.initialize()

        assert runtime.tools == []
        result = await runtime.execute_tool_call(
            {
                "id": "call_ctx",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_context",
                    "arguments": "{}",
                },
            }
        )
        assert "not mapped to an active MCP client" in result

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_message_tool_dispatches_to_shared_service(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(name="builtin-core", transport="builtin", enabled=True),
                ]
            )
        )

        await runtime.initialize()

        with patch(
            "llm_expose.core.builtin_mcp.dispatch_channel_message",
            AsyncMock(
                return_value={
                    "status": "sent",
                    "message_id": "99",
                    "user_id": "84",
                }
            ),
        ) as dispatch_mock:
            result = await runtime.execute_tool_call(
                {
                    "id": "call_send",
                    "type": "function",
                    "function": {
                        "name": "llm_expose_send_message",
                        "arguments": json.dumps(
                            {
                                "target_user_id": "84",
                                "text": "Hello from MCP",
                            }
                        ),
                    },
                },
                execution_context=ToolExecutionContext(
                    execution_mode="chat",
                    channel_id="42",
                    channel_name="support",
                    subject_id="42",
                    subject_kind="user",
                    initiator_user_id="42",
                    platform="telegram",
                    chat_type="private",
                ),
            )

        dispatch_mock.assert_awaited_once_with("support", "84", "Hello from MCP")
        payload = json.loads(result)
        assert payload["status"] == "sent"
        assert payload["message_id"] == "99"
        assert payload["user_id"] == "84"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_message_tool_rejects_unpaired_recipient(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(name="builtin-core", transport="builtin", enabled=True),
                ]
            )
        )

        await runtime.initialize()

        with patch(
            "llm_expose.core.builtin_mcp.dispatch_channel_message",
            AsyncMock(
                side_effect=OutboundMessagePermissionError(
                    "User '84' is not paired with channel 'support'."
                )
            ),
        ):
            result = await runtime.execute_tool_call(
                {
                    "id": "call_send",
                    "type": "function",
                    "function": {
                        "name": "llm_expose_send_message",
                        "arguments": json.dumps(
                            {
                                "target_user_id": "84",
                                "text": "Hello from MCP",
                            }
                        ),
                    },
                },
                execution_context=ToolExecutionContext(
                    execution_mode="chat",
                    channel_id="42",
                    channel_name="support",
                    subject_id="42",
                    subject_kind="user",
                    initiator_user_id="42",
                    platform="telegram",
                    chat_type="private",
                ),
            )

        payload = json.loads(result)
        assert payload["status"] == "error"
        assert payload["channel_name"] == "support"
        assert payload["target_user_id"] == "84"
        assert "not paired" in payload["error"]

        await runtime.shutdown()