"""Tests for builtin MCP runtime behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

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
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )

        await runtime.initialize()

        tool_names = {tool["function"]["name"] for tool in runtime.tools}
        assert tool_names == {
            "llm_expose_get_invocation_context",
            "llm_expose_get_invocation_attachments",
            "llm_expose_get_pairing_ids",
            "llm_expose_send_text_message",
            "llm_expose_send_file_message",
            "llm_expose_send_image_message",
        }

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
    async def test_builtin_get_invocation_attachments_returns_context_attachments(
        self,
    ) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            attachments=[
                {
                    "kind": "image",
                    "source_type": "local_path",
                    "filename": "frame.jpg",
                    "path": "C:/tmp/frame.jpg",
                }
            ],
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_get_attachments",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_attachments",
                    "arguments": "{}",
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["filename"] == "frame.jpg"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_get_invocation_attachments_returns_empty_without_context(
        self,
    ) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        result = await runtime.execute_tool_call(
            {
                "id": "call_get_attachments",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_invocation_attachments",
                    "arguments": "{}",
                },
            },
        )

        payload = json.loads(result)
        assert payload["attachments"] == []

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
    async def test_builtin_send_text_message_tool_executes_with_sender(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        sender.send_message = AsyncMock(
            return_value={"status": "sent", "message_id": "7"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps(
                        {"channel_id": "42", "user_id": "9001", "text": "hello"}
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_message.assert_awaited_once_with("9001", "hello")
        assert payload["status"] == "ok"
        assert payload["tool"] == "llm_expose_send_text_message"
        assert payload["user_id"] == "9001"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_file_message_tool_executes_with_sender(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        test_file = tmp_path / "sample.txt"
        test_file.write_text("hello", encoding="utf-8")

        sender = AsyncMock()
        sender.send_file = AsyncMock(
            return_value={"status": "sent", "file_name": "sample.txt"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_file",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_file_message",
                    "arguments": json.dumps(
                        {
                            "channel_id": "42",
                            "user_id": "9001",
                            "file_path": str(test_file),
                        }
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_file.assert_awaited_once()
        sender.send_file.assert_awaited_once_with("9001", str(test_file.resolve()))
        assert payload["status"] == "ok"
        assert payload["tool"] == "llm_expose_send_file_message"
        assert payload["user_id"] == "9001"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_image_message_tool_executes_with_sender(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        image_file = tmp_path / "snapshot.jpg"
        image_file.write_bytes(b"fake-jpeg")

        sender = AsyncMock()
        sender.send_images = AsyncMock(return_value={"status": "sent", "count": 1})
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_image",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_image_message",
                    "arguments": json.dumps(
                        {
                            "channel_id": "42",
                            "user_id": "9001",
                            "image_path": str(image_file),
                        }
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_images.assert_awaited_once()
        assert sender.send_images.await_args.args[0] == "9001"
        sent_images = sender.send_images.await_args.args[1]
        assert isinstance(sent_images, list)
        assert sent_images[0].startswith("data:image/jpeg;base64,")
        assert payload["status"] == "ok"
        assert payload["tool"] == "llm_expose_send_image_message"
        assert payload["user_id"] == "9001"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_image_message_tool_resolves_attachment_ref(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        image_file = tmp_path / "snapshot.jpg"
        image_file.write_bytes(b"fake-jpeg")

        sender = AsyncMock()
        sender.send_images = AsyncMock(return_value={"status": "sent", "count": 1})
        execution_context = ToolExecutionContext(
            execution_mode="one-shot",
            channel_id="42",
            channel_name="my-bot",
            attachment_paths_by_ref={"att_1": str(image_file.resolve())},
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_image_ref",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_image_message",
                    "arguments": json.dumps(
                        {
                            "channel_id": "42",
                            "user_id": "9001",
                            "attachment_ref": "att_1",
                        }
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_images.assert_awaited_once()
        assert payload["status"] == "ok"
        assert payload["tool"] == "llm_expose_send_image_message"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_tools_return_error_when_sender_missing(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            sender=None,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps(
                        {"channel_id": "42", "user_id": "9001", "text": "hello"}
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        assert payload["status"] == "error"
        assert "sender unavailable" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_file_message_returns_error_for_missing_file(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_file",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_file_message",
                    "arguments": json.dumps(
                        {
                            "channel_id": "42",
                            "user_id": "9001",
                            "file_path": "missing-file.pdf",
                        }
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_file.assert_not_called()
        assert payload["status"] == "error"
        assert "file not found" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_file_message_tool_resolves_attachment_ref(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        test_file = tmp_path / "sample.txt"
        test_file.write_text("hello", encoding="utf-8")

        sender = AsyncMock()
        sender.send_file = AsyncMock(
            return_value={"status": "sent", "file_name": "sample.txt"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="one-shot",
            channel_id="42",
            channel_name="my-bot",
            attachment_paths_by_ref={"att_1": str(test_file.resolve())},
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_file_ref",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_file_message",
                    "arguments": json.dumps(
                        {
                            "channel_id": "42",
                            "user_id": "9001",
                            "attachment_ref": "att_1",
                        }
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_file.assert_awaited_once_with("9001", str(test_file.resolve()))
        assert payload["status"] == "ok"
        assert payload["tool"] == "llm_expose_send_file_message"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_image_message_requires_path_or_ref(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        sender.send_images = AsyncMock(return_value={"status": "sent", "count": 1})
        execution_context = ToolExecutionContext(
            execution_mode="one-shot",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_image_missing",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_image_message",
                    "arguments": json.dumps({"channel_id": "42", "user_id": "9001"}),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_images.assert_not_called()
        assert payload["status"] == "error"
        assert "image_path or attachment_ref" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_get_pairing_ids_returns_configured_pairs(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        pairs_config = tmp_path / "pairs.yaml"
        pairs_config.write_text(
            "pairs_by_channel:\n  my-bot:\n    - '111'\n    - '222'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="111",
            channel_name="my-bot",
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_get_pairs",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_pairing_ids",
                    "arguments": "{}",
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        assert payload["channel_name"] == "my-bot"
        assert payload["pairing_ids"] == ["111", "222"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_get_pairing_ids_error_when_no_channel_name(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name=None,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_get_pairs_no_name",
                "type": "function",
                "function": {
                    "name": "llm_expose_get_pairing_ids",
                    "arguments": "{}",
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        assert payload["status"] == "error"
        assert "channel_name" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_text_message_requires_user_id(self) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text_missing_user",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps({"channel_id": "42", "text": "hello"}),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_message.assert_not_called()
        assert payload["status"] == "error"
        assert "user_id" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_text_message_allows_when_user_id_is_paired(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '9001'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        sender.send_message = AsyncMock(
            return_value={"status": "sent", "message_id": "7"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text_user_paired",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps(
                        {"channel_id": "42", "user_id": "9001", "text": "hello"}
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_message.assert_awaited_once_with("9001", "hello")
        assert payload["status"] == "ok"
        assert payload["user_id"] == "9001"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_text_message_errors_when_user_id_not_paired(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_EXPOSE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "pairs.yaml").write_text(
            "pairs_by_channel:\n  my-bot:\n    - '42'\n",
            encoding="utf-8",
        )

        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        sender.send_message = AsyncMock(
            return_value={"status": "sent", "message_id": "7"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name="my-bot",
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text_unpaired_user",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps(
                        {"channel_id": "42", "user_id": "9001", "text": "hello"}
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_message.assert_not_called()
        assert payload["status"] == "error"
        assert "user_id" in payload["error"]
        assert "not paired" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_builtin_send_text_message_errors_when_channel_name_missing(
        self,
    ) -> None:
        runtime = MCPRuntimeManager(
            MCPConfig(
                servers=[
                    MCPServerConfig(
                        name="builtin-core", transport="builtin", enabled=True
                    ),
                ]
            )
        )
        await runtime.initialize()

        sender = AsyncMock()
        sender.send_message = AsyncMock(
            return_value={"status": "sent", "message_id": "7"}
        )
        execution_context = ToolExecutionContext(
            execution_mode="chat",
            channel_id="42",
            channel_name=None,
            sender=sender,
        )

        result = await runtime.execute_tool_call(
            {
                "id": "call_send_text_missing_channel_name",
                "type": "function",
                "function": {
                    "name": "llm_expose_send_text_message",
                    "arguments": json.dumps(
                        {"channel_id": "42", "user_id": "9001", "text": "hello"}
                    ),
                },
            },
            execution_context=execution_context,
        )

        payload = json.loads(result)
        sender.send_message.assert_not_called()
        assert payload["status"] == "error"
        assert "channel_name" in payload["error"]

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_resets_internal_state_for_reinitialize(self) -> None:
        runtime = MCPRuntimeManager(MCPConfig(servers=[]))
        fake_client = AsyncMock()

        runtime._initialized = True
        runtime._clients = {"server-a": fake_client}
        runtime._tools = [{"type": "function", "function": {"name": "get_files"}}]
        runtime._tool_to_client = {"get_files": fake_client}
        runtime._tool_to_server = {"get_files": "server-a"}
        runtime._server_instructions = ["[server-a] Use get_files for listing"]

        await runtime.shutdown()

        fake_client.close.assert_awaited_once()
        assert runtime._initialized is False
        assert runtime._clients == {}
        assert runtime._tools == []
        assert runtime._tool_to_client == {}
        assert runtime._tool_to_server == {}
        assert runtime._server_instructions == []
