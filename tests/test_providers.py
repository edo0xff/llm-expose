"""Tests for the LiteLLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_expose.config.models import ProviderConfig
from llm_expose.providers.base import BaseProvider
from llm_expose.providers.litellm_provider import LiteLLMProvider


class TestBaseProviderInterface:
    def test_base_provider_is_abstract(self) -> None:
        """BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]


class TestLiteLLMProviderInit:
    def test_is_base_provider(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)
        assert isinstance(provider, BaseProvider)

    def test_sets_api_key_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o", api_key="sk-test")
        LiteLLMProvider(cfg)
        import os

        assert os.environ.get("OPENAI_API_KEY") == "sk-test"

    def test_does_not_override_existing_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "already-set")
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o", api_key="sk-new")
        LiteLLMProvider(cfg)
        import os

        assert os.environ["OPENAI_API_KEY"] == "already-set"

    def test_supports_vision_uses_config_override(self) -> None:
        cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            supports_vision=True,
        )
        provider = LiteLLMProvider(cfg)
        assert provider.supports_vision() is True


class TestModelIdHandling:
    def test_local_model_id_strips_openai_prefix(self) -> None:
        cfg = ProviderConfig(provider_name="local", model="openai/llama3")
        provider = LiteLLMProvider(cfg)
        assert provider._local_model_id() == "llama3"

    def test_non_local_model_is_used_as_configured(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="openai/gpt-4o")
        provider = LiteLLMProvider(cfg)
        kwargs = provider._common_kwargs()
        assert kwargs["model"] == "openai/gpt-4o"


class TestCommonKwargs:
    def test_base_url_included_when_set(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(cfg)
        kwargs = provider._common_kwargs()
        assert kwargs["base_url"] == "http://localhost:1234/v1"

    def test_base_url_absent_when_not_set(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)
        kwargs = provider._common_kwargs()
        assert "base_url" not in kwargs

    def test_model_included_in_common_kwargs(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="openai/gpt-4o")
        provider = LiteLLMProvider(cfg)
        kwargs = provider._common_kwargs()
        assert kwargs["model"] == "openai/gpt-4o"


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello, world!"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_complete_returns_empty_string_when_content_none(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = None

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete([{"role": "user", "content": "Hi"}])

        assert result == ""

    @pytest.mark.asyncio
    async def test_complete_captures_usage_metadata(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.usage.prompt_tokens = 12
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 20
        mock_response.model = "gpt-4o"

        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)),
            patch("litellm.completion_cost", return_value=0.00123),
        ):
            result = await provider.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello, world!"
        usage = provider.get_last_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 8
        assert usage["total_tokens"] == 20
        assert usage["cost_usd"] == 0.00123
        assert usage["model"] == "gpt-4o"
        assert isinstance(usage["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_complete_local_uses_openai_sdk(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "local reply"

        with patch("llm_expose.providers.litellm_provider.AsyncOpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = LiteLLMProvider(cfg)
            result = await provider.complete([{"role": "user", "content": "Hi"}])

            assert result == "local reply"
            mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_passes_tools_and_tool_choice_to_litellm(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)
        tools = [
            {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
        ]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Tool-backed reply"

        with patch(
            "litellm.acompletion", new=AsyncMock(return_value=mock_response)
        ) as mocked_completion:
            result = await provider.complete(
                [{"role": "user", "content": "Hi"}],
                tools=tools,
                tool_choice="required",
            )

        assert result == "Tool-backed reply"
        mocked_completion.assert_awaited_once()
        called_kwargs = mocked_completion.await_args.kwargs
        assert called_kwargs["tools"] == tools
        assert called_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_complete_local_passes_tools_and_tool_choice(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )
        tools = [
            {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
        ]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "local tool reply"

        with patch("llm_expose.providers.litellm_provider.AsyncOpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = LiteLLMProvider(cfg)
            result = await provider.complete(
                [{"role": "user", "content": "Hi"}],
                tools=tools,
                tool_choice="required",
            )

            assert result == "local tool reply"
            called_kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert called_kwargs["tools"] == tools
            assert called_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_complete_strips_images_when_model_has_no_vision(self) -> None:
        cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o-mini",
            supports_vision=False,
        )
        provider = LiteLLMProvider(cfg)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "text only"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,AAAA",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        with (
            patch(
                "litellm.acompletion", new=AsyncMock(return_value=mock_response)
            ) as mocked_completion,
            patch("warnings.warn") as warn_mock,
        ):
            result = await provider.complete(messages)

        assert result == "text only"
        warn_mock.assert_called_once()
        sent_messages = mocked_completion.await_args.kwargs["messages"]
        assert sent_messages[0]["content"] == [{"type": "text", "text": "Describe"}]


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)

        # Build an async iterable of mock chunks
        async def _mock_stream():
            for token in ["Hello", ", ", "world", "!"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        with patch("litellm.acompletion", new=AsyncMock(return_value=_mock_stream())):
            chunks = []
            async for chunk in provider.stream([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world", "!"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_content(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)

        async def _mock_stream():
            for token in ["Hi", None, "!"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        with patch("litellm.acompletion", new=AsyncMock(return_value=_mock_stream())):
            chunks = []
            async for chunk in provider.stream([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

        assert chunks == ["Hi", "!"]

    @pytest.mark.asyncio
    async def test_stream_local_uses_openai_sdk(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )

        async def _mock_stream():
            for token in ["local", " ", "stream"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        with patch("llm_expose.providers.litellm_provider.AsyncOpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_stream())

            provider = LiteLLMProvider(cfg)
            chunks = []
            async for chunk in provider.stream([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

            assert chunks == ["local", " ", "stream"]
            mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_passes_tools_and_tool_choice_to_litellm(self) -> None:
        cfg = ProviderConfig(provider_name="openai", model="gpt-4o")
        provider = LiteLLMProvider(cfg)
        tools = [
            {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
        ]

        async def _mock_stream():
            for token in ["A", "B"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        with patch(
            "litellm.acompletion", new=AsyncMock(return_value=_mock_stream())
        ) as mocked_completion:
            chunks = []
            async for chunk in provider.stream(
                [{"role": "user", "content": "Hi"}],
                tools=tools,
                tool_choice="required",
            ):
                chunks.append(chunk)

        assert chunks == ["A", "B"]
        called_kwargs = mocked_completion.await_args.kwargs
        assert called_kwargs["tools"] == tools
        assert called_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_stream_local_passes_tools_and_tool_choice(self) -> None:
        cfg = ProviderConfig(
            provider_name="local",
            model="llama3",
            base_url="http://localhost:1234/v1",
        )
        tools = [
            {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
        ]

        async def _mock_stream():
            for token in ["local", "-", "tool"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        with patch("llm_expose.providers.litellm_provider.AsyncOpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_stream())

            provider = LiteLLMProvider(cfg)
            chunks = []
            async for chunk in provider.stream(
                [{"role": "user", "content": "Hi"}],
                tools=tools,
                tool_choice="required",
            ):
                chunks.append(chunk)

            assert chunks == ["local", "-", "tool"]
            called_kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert called_kwargs["tools"] == tools
            assert called_kwargs["tool_choice"] == "required"
