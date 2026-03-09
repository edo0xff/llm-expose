"""Reusable tool-aware LLM completion handler (auto-execute mode)."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Optional

from llm_expose.config.models import MCPConfig
from llm_expose.core.mcp_runtime import MCPRuntimeManager
from llm_expose.providers.base import BaseProvider, Message, ToolSpec

logger = logging.getLogger(__name__)


class ToolAwareCompletion:
    """Provider-agnostic tool-aware completion with automatic tool execution.
    
    This handler executes all tool calls automatically without approval.
    For approval-based workflows, use Orchestrator's approval handlers.
    
    Usage:
        async with ToolAwareCompletion(...) as handler:
            response = await handler.complete(messages)
    """
    
    def __init__(
        self,
        provider: BaseProvider,
        mcp_config: Optional[MCPConfig] = None,
        mcp_runtime: Optional[MCPRuntimeManager] = None,
        requested_servers: Optional[list[str]] = None,
        timeout_seconds: int = 30,
    ):
        """Initialize tool-aware completion handler.
        
        Args:
            provider: LLM provider (LiteLLM, etc.)
            mcp_config: MCP configuration (creates new runtime if provided)
            mcp_runtime: Existing runtime (reuse if provided, e.g., from Orchestrator)
            requested_servers: Filter to these server names (used with mcp_config)
            timeout_seconds: Tool execution timeout
        
        Note: Provide EITHER mcp_config OR mcp_runtime, not both.
        """
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self._owns_runtime = mcp_runtime is None
        
        if mcp_runtime is not None:
            # Reuse existing runtime (Orchestrator pattern)
            self._mcp_runtime = mcp_runtime
        elif mcp_config is not None:
            # Create new runtime (message command pattern)
            if requested_servers:
                # Filter config to requested servers
                filtered_config = MCPConfig(
                    servers=[s for s in mcp_config.servers if s.name in requested_servers],
                    settings=mcp_config.settings,
                )
                self._mcp_runtime = MCPRuntimeManager(filtered_config)
            else:
                self._mcp_runtime = MCPRuntimeManager(mcp_config)
        else:
            raise ValueError("Must provide either mcp_config or mcp_runtime")
    
    async def __aenter__(self) -> "ToolAwareCompletion":
        """Context manager entry: initialize runtime if we own it."""
        if self._owns_runtime:
            await self._mcp_runtime.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: shutdown runtime if we own it."""
        if self._owns_runtime:
            await self._mcp_runtime.shutdown()
    
    async def complete(
        self,
        messages: list[Message],
        *,
        max_rounds: int = 8,
    ) -> str:
        """Execute tool-aware completion loop (auto-execute all tools).
        
        Args:
            messages: Conversation history (OpenAI format)
            max_rounds: Maximum tool execution rounds
        
        Returns:
            Final assistant response text
        """
        # Clone messages to avoid mutation
        history = messages[:]
        tools = self._mcp_runtime.tools
        
        for round_num in range(max_rounds):
            # Call LLM with tools
            assistant_message = await self._provider_complete_message(
                history,
                tools=tools,
                tool_choice="auto",
            )
            history.append(assistant_message)
            
            # Check for tool calls
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                # No more tools—return final response
                content = assistant_message.get("content") or ""
                return str(content)
            
            # Execute all tool calls
            await self._execute_tool_calls(history, tool_calls)
        
        # Max rounds exceeded
        fallback = "Tool execution exceeded maximum rounds; stopping to avoid infinite loop."
        history.append({"role": "assistant", "content": fallback})
        return fallback
    
    async def _provider_complete_message(
        self,
        history: list[Message],
        *,
        tools: list[ToolSpec],
        tool_choice: str,
    ) -> Message:
        """Call provider with tools, returning Message dict.
        
        Mirrors Orchestrator._provider_complete_message pattern:
        - Try complete_with_message() first (returns Message with tool_calls)
        - Fallback to complete() if not available
        """
        complete_with_message = getattr(self._provider, "complete_with_message", None)
        if callable(complete_with_message):
            maybe_message = complete_with_message(
                history,
                tools=tools,
                tool_choice=tool_choice,
            )
            if inspect.isawaitable(maybe_message):
                message = await maybe_message
                if isinstance(message, dict):
                    return message
                if hasattr(message, "model_dump"):
                    return message.model_dump(exclude_none=True)
                return {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", ""),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
        
        # Fallback to simple complete()
        content = await self._provider.complete(
            history,
            tools=tools,
            tool_choice=tool_choice,
        )
        return {"role": "assistant", "content": content}
    
    async def _execute_tool_calls(
        self,
        history: list[Message],
        tool_calls: list[Any],
    ) -> None:
        """Execute tool calls and append results to history.
        
        Mirrors Orchestrator._execute_tool_calls pattern.
        """
        for call in tool_calls:
            call_id = self._tool_call_id(call)
            try:
                tool_result = await asyncio.wait_for(
                    self._mcp_runtime.execute_tool_call(call),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                tool_result = f"MCP tool execution timed out after {self._timeout_seconds} seconds."
            except Exception as exc:
                tool_result = f"MCP tool execution failed: {exc}"
            
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result,
            })
    
    @staticmethod
    def _tool_call_id(tool_call: Any) -> str:
        """Extract tool call ID (OpenAI format).
        
        Mirrors Orchestrator._tool_call_id static method.
        """
        if isinstance(tool_call, dict):
            return str(tool_call.get("id") or "unknown_tool_call")
        return str(getattr(tool_call, "id", "unknown_tool_call"))
