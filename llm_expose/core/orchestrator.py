"""Orchestrator: ties together an LLM provider and a messaging client."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from llm_expose.clients.base import BaseClient, MessageResponse
from llm_expose.config.loader import get_pairs_for_channel, load_mcp_config
from llm_expose.config.models import ExposureConfig, MCPSettingsConfig
from llm_expose.core.mcp_runtime import MCPRuntimeManager
from llm_expose.providers.base import BaseProvider, Message, ToolChoice, ToolSpec

logger = logging.getLogger(__name__)

# System prompt injected at the start of every conversation.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
)


@dataclass
class _PendingApproval:
    approval_id: str
    created_at: float
    tools: list[ToolSpec]
    tool_calls: list[Any]
    server_names: dict[str, str]  # tool_name -> server_name mapping


class Orchestrator:
    """Connects an LLM *provider* to a messaging *client*.

    The orchestrator maintains per-channel conversation history so that the
    LLM retains context across multiple turns in a single chat session. Each
    channel can have its own system prompt to provide context about the
    communication channel being used.

    Args:
        config: The :class:`~llm_expose.config.models.ExposureConfig` that
            describes both the provider and the client to use.
        provider: An initialised :class:`~llm_expose.providers.base.BaseProvider`.
        client: An initialised :class:`~llm_expose.clients.base.BaseClient`.
    """

    def __init__(
        self,
        config: ExposureConfig,
        provider: BaseProvider,
        client: BaseClient,
    ) -> None:
        self._config = config
        self._provider = provider
        self._client = client
        # Per-channel conversation histories (keyed by channel ID).
        # Each channel gets its own history with the configured system prompt.
        self._histories: dict[str, list[Message]] = {}
        # Use custom system prompt from channel config, or fall back to default.
        self._system_prompt = config.client.system_prompt or _DEFAULT_SYSTEM_PROMPT
        # Backward compatibility for older tests/callers that access _history
        # or call _handle_message(user_message) without a channel ID.
        self._default_channel_id = "__default__"
        self._history = self._get_or_create_history(self._default_channel_id)
        self._tool_choice: ToolChoice | None = "auto"
        self._mcp_runtime: MCPRuntimeManager | None = None
        self._mcp_runtime_initialized = False
        self._mcp_runtime_lock = asyncio.Lock()
        self._mcp_settings = MCPSettingsConfig()
        self._pending_approvals: dict[str, _PendingApproval] = {}
        self._approval_ttl_seconds = 600
        self._channel_name = config.channel_name
        self._paired_channel_ids: set[str] = set()

        if self._channel_name:
            try:
                self._paired_channel_ids = set(get_pairs_for_channel(self._channel_name))
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.warning("Failed to load channel pairs: %s", exc)

        try:
            mcp_config = load_mcp_config()
            self._mcp_settings = mcp_config.settings
            attached_server_names = set(config.client.mcp_servers)
            if attached_server_names:
                available_server_names = {server.name for server in mcp_config.servers}
                missing_server_names = sorted(attached_server_names - available_server_names)
                if missing_server_names:
                    logger.warning(
                        "Channel '%s' has missing MCP server attachments: %s",
                        config.name,
                        ", ".join(missing_server_names),
                    )

                mcp_config.servers = [
                    server
                    for server in mcp_config.servers
                    if server.name in attached_server_names
                ]
            else:
                mcp_config.servers = []

            if any(server.enabled for server in mcp_config.servers):
                self._mcp_runtime = MCPRuntimeManager(mcp_config)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Failed to load MCP config: %s", exc)

    async def _ensure_mcp_runtime_ready(self) -> None:
        if self._mcp_runtime is None or self._mcp_runtime_initialized:
            return
        async with self._mcp_runtime_lock:
            if self._mcp_runtime_initialized:
                return
            await self._mcp_runtime.initialize()
            self._mcp_runtime_initialized = True

    def _get_or_create_history(self, channel_id: str) -> list[Message]:
        """Get the conversation history for a channel, creating it if needed.

        Args:
            channel_id: Unique identifier for the channel/chat.

        Returns:
            The conversation history list for this channel.
        """
        if channel_id not in self._histories:
            logger.debug("Creating new conversation history for channel %s", channel_id)

            # Build system prompt with MCP server instructions
            system_content = self._system_prompt

            # Add MCP server instructions if runtime is available and initialized
            # Use getattr to handle cases where _mcp_runtime may not be set yet (during __init__)
            mcp_runtime = getattr(self, "_mcp_runtime", None)
            mcp_runtime_initialized = getattr(self, "_mcp_runtime_initialized", False)

            if mcp_runtime and mcp_runtime_initialized:
                mcp_instructions = mcp_runtime.server_instructions
                if mcp_instructions:
                    system_content += "\n\n## These are the available MCP tools that you can call:\n\n" + mcp_instructions
                    system_content += "\n\nTo call a tool, you must output only a JSON object in this format:"
                    system_content += "\n\n{\"name\": \"tool_name\", \"args\": {\"arg_name\": \"value\"}}"
                    system_content += "\n\nIf no tool is needed, respond normally."

            self._histories[channel_id] = [
                {"role": "system", "content": system_content}
            ]
        return self._histories[channel_id]

    async def _handle_message(
        self, channel_or_message: str, user_message: str | None = None
    ) -> str | MessageResponse:
        """Process a single user message and return the LLM's reply.

        Appends the user message to the channel's history, calls the provider,
        then appends the assistant's reply and returns it.

        Args:
            channel_or_message: Either channel ID (new style) or user message
                (legacy one-argument style).
            user_message: Raw text from the end user when channel ID is passed.

        Returns:
            The LLM's reply as a plain string or MessageResponse with approval metadata.
        """
        if user_message is None:
            channel_id = self._default_channel_id
            text = channel_or_message
        else:
            channel_id = channel_or_message
            text = user_message

        if not self._is_channel_paired(channel_id):
            return f"This instance is not paired. Run llm-expose add pair {channel_id}"

        approval_decision = self._parse_approval_decision(text)
        if approval_decision is not None:
            return await self._handle_approval_decision(channel_id, *approval_decision)

        if channel_id in self._pending_approvals:
            pending = self._pending_approvals[channel_id]
            if self._is_pending_expired(pending):
                del self._pending_approvals[channel_id]
            else:
                return (
                    f"A tool call is waiting for confirmation (id: {pending.approval_id}). "
                    f"Reply `approve {pending.approval_id}` or `reject {pending.approval_id}`."
                )

        history = self._get_or_create_history(channel_id)
        history.append({"role": "user", "content": text})
        logger.debug(
            "Sending %d messages to provider for channel %s",
            len(history),
            channel_id,
        )

        await self._ensure_mcp_runtime_ready()
        tools = self._mcp_runtime.tools if self._mcp_runtime is not None else []
        if not tools:
            reply = await self._provider.complete(history)
            history.append({"role": "assistant", "content": reply})
            return reply

        # Use mixed approval handler which respects per-server confirmation settings
        return await self._handle_message_with_mixed_approval(history, tools, channel_id)

    def _is_channel_paired(self, channel_id: str) -> bool:
        """Return whether an incoming channel ID is allowed for this exposure.

        Pair enforcement is active only when a channel namespace was supplied in
        config. This keeps backward compatibility for direct programmatic usage.
        """
        if not self._channel_name:
            return True

        # Reload persisted pair data on each check so runtime add/remove
        # operations are reflected without restarting the exposure process.
        try:
            self._paired_channel_ids = set(get_pairs_for_channel(self._channel_name))
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Failed to refresh channel pairs: %s", exc)
            return False

        return channel_id in self._paired_channel_ids

    @staticmethod
    def _parse_approval_decision(text: str) -> tuple[bool, str] | None:
        parts = text.strip().split(maxsplit=1)
        if len(parts) != 2:
            return None
        command = parts[0].lower()
        approval_id = parts[1].strip()
        if command == "approve":
            return True, approval_id
        if command == "reject":
            return False, approval_id
        return None

    def _is_pending_expired(self, pending: _PendingApproval) -> bool:
        return (time.monotonic() - pending.created_at) > self._approval_ttl_seconds

    def _get_tool_confirmation_mode(self, tool_name: str) -> str:
        """Determine confirmation mode for a tool based on its server config.
        
        Returns:
            "required" if tool needs approval, "optional" if it auto-executes.
        """
        if self._mcp_runtime is None:
            return self._mcp_settings.confirmation_mode
        
        server_name = self._mcp_runtime.get_tool_server_name(tool_name)
        if server_name is None:
            return self._mcp_settings.confirmation_mode
        
        server_config = self._mcp_runtime.get_server_config(server_name)
        if server_config is None:
            return self._mcp_settings.confirmation_mode
        
        # Resolve per-server confirmation setting
        if server_config.tool_confirmation == "default":
            return self._mcp_settings.confirmation_mode
        elif server_config.tool_confirmation == "required":
            return "required"
        else:  # "never"
            return "optional"

    async def _handle_message_with_required_approval(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        channel_id: str,
    ) -> str:
        assistant_message = await self._provider_complete_message(
            history,
            tools=tools,
            tool_choice=self._tool_choice,
        )
        history.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return str(assistant_message.get("content") or "")

        # Build server name mapping for the tool calls
        server_names: dict[str, str] = {}
        if self._mcp_runtime is not None:
            for call in tool_calls:
                function_obj = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                if isinstance(function_obj, dict):
                    tool_name = function_obj.get("name")
                else:
                    tool_name = getattr(function_obj, "name", None)
                if tool_name:
                    server_name = self._mcp_runtime.get_tool_server_name(tool_name)
                    if server_name:
                        server_names[tool_name] = server_name

        approval_id = uuid.uuid4().hex[:8]
        self._pending_approvals[channel_id] = _PendingApproval(
            approval_id=approval_id,
            created_at=time.monotonic(),
            tools=tools,
            tool_calls=tool_calls,
            server_names=server_names,
        )
        return self._format_approval_prompt(approval_id, tool_calls, server_names)

    def _format_approval_prompt(self, approval_id: str, tool_calls: list[Any], server_names: dict[str, str] | None = None) -> MessageResponse:
        tool_info: list[str] = []
        tool_names: list[str] = []
        for call in tool_calls:
            function_obj = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            if isinstance(function_obj, dict):
                name = function_obj.get("name")
            else:
                name = getattr(function_obj, "name", None)
            tool_name = str(name or "unknown_tool")
            tool_names.append(tool_name)
            
            # Add server name if available
            if server_names and tool_name in server_names:
                tool_info.append(f"{tool_name} ({server_names[tool_name]})")
            else:
                tool_info.append(tool_name)

        names = ", ".join(tool_info)
        content = (
            f"Tool execution requires confirmation (id: {approval_id}).\n"
            f"Requested tool(s): {names}\n"
            f"Reply `approve {approval_id}` to continue or `reject {approval_id}` to cancel."
        )
        return MessageResponse(
            content=content,
            approval_id=approval_id,
            tool_names=tool_names,
            server_names=server_names or {}
        )

    async def _handle_message_with_mixed_approval(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        channel_id: str,
    ) -> str:
        """Handle message with per-server tool confirmation settings.
        
        This method splits tool calls into auto-execute and needs-approval groups,
        executes auto-execute tools immediately, and creates pending approvals for
        tools that require confirmation.
        """
        max_tool_rounds = 8

        for _ in range(max_tool_rounds):
            assistant_message = await self._provider_complete_message(
                history,
                tools=tools,
                tool_choice=self._tool_choice,
            )
            history.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return str(assistant_message.get("content") or "")

            # Split tool calls by confirmation requirement
            auto_execute_calls: list[Any] = []
            needs_approval_calls: list[Any] = []

            for call in tool_calls:
                function_obj = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                if isinstance(function_obj, dict):
                    tool_name = function_obj.get("name")
                else:
                    tool_name = getattr(function_obj, "name", None)

                if tool_name:
                    confirmation_mode = self._get_tool_confirmation_mode(tool_name)
                    if confirmation_mode == "required":
                        needs_approval_calls.append(call)
                    else:
                        auto_execute_calls.append(call)
                else:
                    # Unknown tool name, default to auto-execute
                    auto_execute_calls.append(call)

            # Execute auto-execute tools immediately
            if auto_execute_calls:
                if self._mcp_runtime is None:
                    return "MCP runtime unavailable while handling tool calls."
                await self._execute_tool_calls(history, auto_execute_calls)

            # If there are tools needing approval, create pending approval
            if needs_approval_calls:
                # Build server name mapping for the tool calls needing approval
                server_names: dict[str, str] = {}
                if self._mcp_runtime is not None:
                    for call in needs_approval_calls:
                        function_obj = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                        if isinstance(function_obj, dict):
                            tool_name = function_obj.get("name")
                        else:
                            tool_name = getattr(function_obj, "name", None)
                        if tool_name:
                            server_name = self._mcp_runtime.get_tool_server_name(tool_name)
                            if server_name:
                                server_names[tool_name] = server_name

                approval_id = uuid.uuid4().hex[:8]
                self._pending_approvals[channel_id] = _PendingApproval(
                    approval_id=approval_id,
                    created_at=time.monotonic(),
                    tools=tools,
                    tool_calls=needs_approval_calls,
                    server_names=server_names,
                )
                return self._format_approval_prompt(approval_id, needs_approval_calls, server_names)

            # If only auto-execute tools were present, continue the loop

        fallback = "Tool execution exceeded maximum rounds; stopping to avoid infinite loop."
        history.append({"role": "assistant", "content": fallback})
        return fallback

    async def _handle_approval_decision(
        self,
        channel_id: str,
        approved: bool,
        approval_id: str,
    ) -> str:
        pending = self._pending_approvals.get(channel_id)
        if pending is None:
            return "There is no pending tool approval for this chat."

        if self._is_pending_expired(pending):
            del self._pending_approvals[channel_id]
            return "The pending tool approval expired. Please ask again."

        if pending.approval_id != approval_id:
            return (
                f"Unknown approval id '{approval_id}'. "
                f"Pending id is '{pending.approval_id}'."
            )

        history = self._get_or_create_history(channel_id)
        del self._pending_approvals[channel_id]

        if approved:
            if self._mcp_runtime is None:
                return "MCP runtime unavailable while handling approval."
            await self._execute_tool_calls(history, pending.tool_calls)
            return await self._handle_message_with_mcp_tools(history, pending.tools)

        for call in pending.tool_calls:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": self._tool_call_id(call),
                    "content": "User rejected tool execution.",
                }
            )
        reply = await self._provider.complete(history)
        history.append({"role": "assistant", "content": reply})
        return reply

    async def _execute_tool_calls(
        self,
        history: list[Message],
        tool_calls: list[Any],
    ) -> None:
        if self._mcp_runtime is None:
            return
        for call in tool_calls:
            call_id = self._tool_call_id(call)
            try:
                tool_result = await asyncio.wait_for(
                    self._mcp_runtime.execute_tool_call(call),
                    timeout=self._mcp_settings.tool_timeout_seconds,
                )
            except asyncio.TimeoutError:
                tool_result = (
                    "MCP tool execution timed out after "
                    f"{self._mcp_settings.tool_timeout_seconds} seconds."
                )
            except Exception as exc:
                tool_result = f"MCP tool execution failed: {exc}"
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result,
                }
            )
            

    async def _provider_complete_message(
        self,
        history: list[Message],
        *,
        tools: list[ToolSpec],
        tool_choice: ToolChoice | None,
    ) -> Message:
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

        content = await self._provider.complete(
            history,
            tools=tools,
            tool_choice=tool_choice,
        )
        return {"role": "assistant", "content": content}

    async def _handle_message_with_mcp_tools(
        self,
        history: list[Message],
        tools: list[ToolSpec],
    ) -> str:
        max_tool_rounds = 8

        for _ in range(max_tool_rounds):
            assistant_message = await self._provider_complete_message(
                history,
                tools=tools,
                tool_choice=self._tool_choice,
            )
            history.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return str(assistant_message.get("content") or "")

            if self._mcp_runtime is None:
                return "MCP runtime unavailable while handling tool calls."
            await self._execute_tool_calls(history, tool_calls)

        fallback = "Tool execution exceeded maximum rounds; stopping to avoid infinite loop."
        history.append({"role": "assistant", "content": fallback})
        return fallback

    @staticmethod
    def _tool_call_id(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("id") or "unknown_tool_call")
        return str(getattr(tool_call, "id", "unknown_tool_call"))

    async def run(self) -> None:
        """Start the client and block until it stops.

        Registers :meth:`_handle_message` as the client's message handler and
        delegates lifecycle management to the client.
        """
        # Wire our handler into the client via the public setter
        self._client.set_handler(self._handle_message)
        logger.info("Starting orchestrator for exposure '%s'", self._config.name)
        try:
            await self._client.start()
        finally:
            if self._mcp_runtime is not None:
                await self._mcp_runtime.shutdown()
            # Always attempt a graceful shutdown to release client resources
            # (e.g., Telegram long-polling session) even on Ctrl+C/errors.
            await self._client.stop()
