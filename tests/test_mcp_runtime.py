"""Tests for builtin MCP runtime behavior."""

from __future__ import annotations

import json

import pytest

from llm_expose.config.models import MCPConfig, MCPServerConfig
from llm_expose.core.builtin_mcp import ToolExecutionContext
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