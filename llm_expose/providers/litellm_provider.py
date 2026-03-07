"""LiteLLM-backed provider implementation."""

from __future__ import annotations

import os
from typing import AsyncIterator

import litellm
from openai import AsyncOpenAI

from llm_expose.config.models import ProviderConfig
from llm_expose.providers.base import BaseProvider


class LiteLLMProvider(BaseProvider):
    """LLM provider that delegates to `litellm`.

    LiteLLM supports OpenAI, Anthropic, Google, and many other backends, as
    well as local models that expose an OpenAI-compatible REST API (e.g.
    LM Studio, Ollama with the OpenAI proxy, vLLM).

    Args:
        config: The :class:`~llm_expose.config.models.ProviderConfig` that
            controls which model and settings are used.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._openai_client: AsyncOpenAI | None = None
        if config.api_key:
            # LiteLLM reads API keys from environment variables.  Set the
            # provider-specific variable so callers don't have to manage it.
            env_key = f"{config.provider_name.upper()}_API_KEY"
            os.environ.setdefault(env_key, config.api_key)
        if self._is_local_provider():
            # Most local OpenAI-compatible servers ignore API keys, but the
            # SDK expects one; use a harmless default when omitted.
            self._openai_client = AsyncOpenAI(
                base_url=config.base_url or "http://localhost:1234/v1",
                api_key=config.api_key or "local-not-required",
            )

    def _is_local_provider(self) -> bool:
        """Return ``True`` when configuration targets a local model server."""
        return self._config.provider_name.lower() == "local"

    def _local_model_id(self) -> str:
        """Return the model identifier expected by OpenAI-compatible local APIs."""
        model = self._config.model.strip()
        # Backward compatibility for existing configs saved as openai/<model>.
        if model.startswith("openai/"):
            return model.split("/", 1)[1]
        return model

    def _build_model_id(self) -> str:
        """Return the fully-qualified model identifier expected by LiteLLM.

        For *local* providers the caller is expected to pass an already-qualified
        string such as ``openai/llama3``; for well-known online providers the
        ``provider_name/model`` form is used.
        """
        name = self._config.provider_name.lower()
        model = self._config.model
        if name == "local":
            # Local providers are handled by the OpenAI SDK path.
            return self._local_model_id()
        # If model already contains a slash it is already qualified.
        if "/" in model:
            return model
        return f"{name}/{model}"

    def _common_kwargs(self) -> dict:
        """Build the kwargs shared by both sync and streaming calls."""
        kwargs: dict = {
            "model": self._build_model_id(),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        return kwargs

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a single completion from the configured model.

        Args:
            messages: OpenAI-style message list.

        Returns:
            The model's reply as a plain string.

        Raises:
            litellm.exceptions.APIError: On provider-level errors.
        """
        if self._is_local_provider():
            assert self._openai_client is not None
            response = await self._openai_client.chat.completions.create(
                model=self._local_model_id(),
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            return response.choices[0].message.content or ""

        response = await litellm.acompletion(
            messages=messages,
            **self._common_kwargs(),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield text chunks from a streaming completion.

        Args:
            messages: OpenAI-style message list.

        Yields:
            Text delta strings as they are received from the model.
        """
        if self._is_local_provider():
            assert self._openai_client is not None
            response = await self._openai_client.chat.completions.create(
                model=self._local_model_id(),
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        response = await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._common_kwargs(),
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
