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


class TestBuildModelId:
    def _provider(self, provider_name: str, model: str) -> LiteLLMProvider:
        cfg = ProviderConfig(provider_name=provider_name, model=model)
        return LiteLLMProvider(cfg)

    def test_online_provider_prefixed(self) -> None:
        p = self._provider("openai", "gpt-4o")
        assert p._build_model_id() == "openai/gpt-4o"

    def test_local_provider_uses_openai_prefix(self) -> None:
        p = self._provider("local", "llama3")
        assert p._build_model_id() == "openai/llama3"

    def test_already_qualified_model_unchanged(self) -> None:
        p = self._provider("openai", "openai/gpt-4o")
        assert p._build_model_id() == "openai/gpt-4o"

    def test_anthropic_provider(self) -> None:
        p = self._provider("anthropic", "claude-3-5-sonnet")
        assert p._build_model_id() == "anthropic/claude-3-5-sonnet"


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

    def test_temperature_and_max_tokens(self) -> None:
        cfg = ProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=512,
        )
        provider = LiteLLMProvider(cfg)
        kwargs = provider._common_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 512


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
