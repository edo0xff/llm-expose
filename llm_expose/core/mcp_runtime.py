"""Runtime MCP bridge utilities using FastMCP client."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from llm_expose.config.models import MCPConfig, MCPServerConfig
from llm_expose.core.builtin_mcp import BuiltinMCPClient, ToolExecutionContext
from llm_expose.core.content_parts import content_has_images, normalize_mcp_content
from llm_expose.providers.base import ToolSpec

logger = logging.getLogger(__name__)


class MCPRuntimeManager:
    """Manage MCP sessions and tool execution using FastMCP client.

    This runtime uses the FastMCP client library for stable MCP protocol support
    across stdio, SSE, and HTTP transports.
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._stack = AsyncExitStack()
        self._initialized = False
        self._tools: list[ToolSpec] = []
        self._tool_to_client: dict[str, Any] = {}  # tool_name -> fastmcp.Client
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._clients: dict[str, Any] = {}  # server_name -> fastmcp.Client
        self._server_instructions: list[str] = []  # Aggregated instructions from servers

    @property
    def tools(self) -> list[ToolSpec]:
        """Return discovered OpenAI-compatible tool definitions."""
        return self._tools

    @property
    def server_instructions(self) -> str:
        """Return aggregated server instructions for the system prompt."""
        if not self._server_instructions:
            return ""
        return "\n\n".join(self._server_instructions)

    def get_tool_server_name(self, tool_name: str) -> str | None:
        """Return the name of the server that provides the given tool."""
        return self._tool_to_server.get(tool_name)

    def get_server_config(self, server_name: str) -> MCPServerConfig | None:
        """Return the configuration for the given server."""
        for server in self._config.servers:
            if server.name == server_name:
                return server
        return None

    async def initialize(self) -> None:
        """Open MCP connections and load tools from enabled servers."""
        if self._initialized:
            return

        client_cls: Any | None = None
        stdio_transport_cls: Any | None = None
        sse_transport_cls: Any | None = None
        http_transport_cls: Any | None = None

        needs_external_runtime = any(
            server.enabled and server.transport != "builtin"
            for server in self._config.servers
        )
        if needs_external_runtime:
            try:
                from fastmcp import Client
                from fastmcp.client.transports import (
                    SSETransport,
                    StdioTransport,
                    StreamableHttpTransport,
                )

                client_cls = Client
                stdio_transport_cls = StdioTransport
                sse_transport_cls = SSETransport
                http_transport_cls = StreamableHttpTransport
            except Exception as exc:
                logger.warning(
                    "FastMCP unavailable. External MCP servers will be skipped. Error: %s",
                    exc,
                )

        for server in self._config.servers:
            if not server.enabled:
                continue
            try:
                client = await self._create_client(
                    server,
                    client_cls=client_cls,
                    stdio_transport_cls=stdio_transport_cls,
                    sse_transport_cls=sse_transport_cls,
                    http_transport_cls=http_transport_cls,
                )
                if client is None:
                    continue

                # Initialize the client connection
                await self._stack.enter_async_context(client)

                # Store client reference
                self._clients[server.name] = client

                # Discover tools from the server
                mcp_tools = await client.list_tools()
                self._register_server_tools(server, client, mcp_tools)

                # Fetch server prompts/instructions for tool usage guidance
                await self._fetch_server_prompts(server, client)

            except Exception as exc:
                logger.warning("Failed to initialize MCP server '%s': %s", server.name, exc)

        self._initialized = True

    async def _create_client(
        self,
        server: MCPServerConfig,
        client_cls: Any,
        stdio_transport_cls: Any,
        sse_transport_cls: Any,
        http_transport_cls: Any,
    ) -> Any | None:
        """Create a FastMCP client for the given server configuration."""
        if server.transport == "builtin":
            return BuiltinMCPClient(server_name=server.name)

        if client_cls is None:
            logger.warning(
                "Skipping MCP server '%s': external MCP runtime is unavailable.",
                server.name,
            )
            return None

        if server.transport == "stdio":
            if not server.command:
                logger.warning(
                    "Skipping stdio MCP server '%s': missing command.",
                    server.name,
                )
                return None

            transport = stdio_transport_cls(
                command=server.command,
                args=server.args,
                env=server.env or {},
            )
            return client_cls(transport)

        if server.transport == "sse":
            if not server.url:
                logger.warning(
                    "Skipping SSE MCP server '%s': missing URL.",
                    server.name,
                )
                return None

            transport = sse_transport_cls(url=server.url)
            return client_cls(transport)

        if server.transport == "http":
            if not server.url:
                logger.warning(
                    "Skipping HTTP MCP server '%s': missing URL.",
                    server.name,
                )
                return None

            transport = http_transport_cls(url=server.url)
            return client_cls(transport)

        logger.warning(
            "Skipping MCP server '%s': unsupported transport '%s'.",
            server.name,
            server.transport,
        )
        return None

    def _register_server_tools(
        self,
        server: MCPServerConfig,
        client: Any,
        mcp_tools: Any,
    ) -> None:
        """Convert MCP tools to OpenAI format and register routing."""
        allowed = set(server.allowed_tools)

        # mcp_tools could be a list or a response object with .tools attribute
        tools_list = mcp_tools if isinstance(mcp_tools, list) else getattr(mcp_tools, "tools", [])

        for tool in tools_list:
            # Convert MCP tool to OpenAI-compatible format
            openai_tool = self._mcp_tool_to_openai(tool)
            if not openai_tool:
                continue

            tool_name = openai_tool["function"]["name"]

            # Apply allowlist filter
            if allowed and tool_name not in allowed:
                continue

            self._tools.append(openai_tool)
            self._tool_to_client[tool_name] = client
            self._tool_to_server[tool_name] = server.name

    async def _fetch_server_prompts(self, server: MCPServerConfig, client: Any) -> None:
        """Fetch and store server prompts/instructions for tool usage guidance."""
        try:
            # List available prompts from the server
            prompts_result = await client.list_prompts()
            prompts_list = prompts_result if isinstance(prompts_result, list) else getattr(prompts_result, "prompts", [])

            if not prompts_list:
                logger.debug("No prompts available from MCP server '%s'", server.name)
                return

            # Fetch each prompt's content
            for prompt in prompts_list:
                try:
                    prompt_name = prompt if isinstance(prompt, str) else getattr(prompt, "name", None)
                    if not prompt_name:
                        continue

                    # Get the prompt content
                    prompt_result = await client.get_prompt(prompt_name)
                    prompt_dict = self._to_dict(prompt_result)

                    # Extract message content from the prompt
                    messages = prompt_dict.get("messages", [])
                    for msg in messages:
                        msg_dict = self._to_dict(msg) if not isinstance(msg, dict) else msg
                        content = msg_dict.get("content", {}) if isinstance(msg_dict.get("content"), dict) else msg_dict.get("content")

                        # Handle both string and structured content
                        if isinstance(content, str) and content.strip():
                            instruction = f"[{server.name}] {content.strip()}"
                            self._server_instructions.append(instruction)
                        elif isinstance(content, dict):
                            text = content.get("text", "")
                            if text and isinstance(text, str) and text.strip():
                                instruction = f"[{server.name}] {text.strip()}"
                                self._server_instructions.append(instruction)

                except Exception as prompt_exc:
                    logger.debug("Could not fetch prompt '%s' from server '%s': %s", prompt_name, server.name, prompt_exc)

        except Exception as exc:
            logger.debug("No prompts available from MCP server '%s': %s", server.name, exc)

    def _mcp_tool_to_openai(self, mcp_tool: Any) -> dict[str, Any] | None:
        """Convert MCP tool schema to OpenAI function tool format."""
        try:
            # MCP tools have: name, description, inputSchema
            tool_dict = self._to_dict(mcp_tool)

            name = tool_dict.get("name")
            if not name or not isinstance(name, str):
                return None

            description = tool_dict.get("description", "")
            input_schema = tool_dict.get("inputSchema", {})

            # OpenAI function tool format
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema or {"type": "object", "properties": {}},
                },
            }
        except Exception as exc:
            logger.warning("Failed to convert MCP tool to OpenAI format: %s", exc)
            return None

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        """Convert various types to dictionary."""
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if hasattr(value, "dict"):
            return value.dict(exclude_none=True)
        try:
            return dict(value)
        except Exception:
            return {"value": str(value)}

    async def execute_tool_call(
        self,
        openai_tool_call: Any,
        *,
        execution_context: ToolExecutionContext | None = None,
    ) -> str | list[dict[str, Any]]:
        """Execute a single OpenAI tool call and return text or structured content."""
        logger.info("Received tool call: %s", openai_tool_call)
        tool_call_dict = self._to_dict(openai_tool_call)

        # Extract tool name and arguments from OpenAI format
        function_obj = tool_call_dict.get("function", {})
        tool_name = function_obj.get("name")

        if not tool_name:
            return "MCP tool execution failed: missing tool name in model tool call payload."

        client = self._tool_to_client.get(tool_name)
        if client is None:
            return f"MCP tool execution failed: tool '{tool_name}' is not mapped to an active MCP client."

        try:
            # Parse arguments (could be string or dict)
            arguments = function_obj.get("arguments", "{}")
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}

            # Call the tool via FastMCP client
            if hasattr(client, "call_tool_with_context"):
                result = await client.call_tool_with_context(
                    tool_name,
                    arguments,
                    execution_context=execution_context,
                )
            else:
                result = await client.call_tool(tool_name, arguments)

            logger.info("Executed tool call '%s' with result: %s", tool_name, result)
            normalized_content = self._result_to_openai_content(result)
            if normalized_content and content_has_images(normalized_content):
                return normalized_content
            return self._result_to_text(result)
        except Exception as exc:
            return f"MCP tool execution failed for '{tool_name}': {exc}"

    @staticmethod
    def _result_to_openai_content(result: Any) -> list[dict[str, Any]]:
        """Convert MCP tool result to OpenAI-compatible content blocks."""
        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = MCPRuntimeManager._to_dict(result)
        content = result_dict.get("content")
        return normalize_mcp_content(content)

    @staticmethod
    def _result_to_text(result: Any) -> str:
        """Convert MCP tool result to plain text."""
        # FastMCP results format: could be string, dict, or list of content items
        if isinstance(result, str):
            return result

        result_dict = MCPRuntimeManager._to_dict(result) if not isinstance(result, dict) else result

        # Check for content list (MCP standard format)
        content = result_dict.get("content")
        if content and isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                    else:
                        parts.append(str(item))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip() or str(result)

        # Fallback to string representation
        return str(result)

    async def shutdown(self) -> None:
        """Close all active MCP client connections."""
        for server_name, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "disconnect"):
                    await client.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting MCP client '%s': %s", server_name, exc)
        await self._stack.aclose()
