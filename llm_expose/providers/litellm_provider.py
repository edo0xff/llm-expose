"""LiteLLM-backed provider implementation."""

from __future__ import annotations

import os
from typing import AsyncIterator

import litellm

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
        if config.api_key:
            # LiteLLM reads API keys from environment variables.  Set the
            # provider-specific variable so callers don't have to manage it.
            env_key = f"{config.provider_name.upper()}_API_KEY"
            os.environ.setdefault(env_key, config.api_key)

    def _build_model_id(self) -> str:
        """Return the fully-qualified model identifier expected by LiteLLM.

        For *local* providers the caller is expected to pass an already-qualified
        string such as ``openai/llama3``; for well-known online providers the
        ``provider_name/model`` form is used.
        """
        name = self._config.provider_name.lower()
        model = self._config.model
        # If model already contains a slash it is already qualified.
        if "/" in model:
            return model
        if name == "local":
            # Local OpenAI-compatible server – use the openai/ prefix so
            # LiteLLM routes to the openai SDK with the custom base_url.
            return f"openai/{model}"
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
        response = await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._common_kwargs(),
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
