"""LiteLLM-backed provider implementation."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import litellm
from openai import AsyncOpenAI

from llm_expose.config.models import ProviderConfig
from llm_expose.providers.base import BaseProvider, Message, ToolChoice, ToolSpec


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
            litellm.api_key = config.api_key
            # Preserve historical behavior expected by tests and local users.
            if (
                config.provider_name.lower() == "openai"
                and not os.environ.get("OPENAI_API_KEY")
            ):
                os.environ["OPENAI_API_KEY"] = config.api_key
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

    def _common_kwargs(self) -> dict:
        """Build the kwargs shared by both sync and streaming calls."""
        kwargs: dict = {
            "model": self._config.model.strip(),
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        return kwargs

    @staticmethod
    def _message_to_dict(message: Any) -> Message:
        """Normalize provider message objects into plain dicts."""
        if isinstance(message, dict):
            return message
        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None),
        }

    async def complete_with_message(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Message:
        """Return the full assistant message payload for tool-call handling."""
        request_kwargs: dict[str, Any] = {}
        if tools is not None:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice

        if self._is_local_provider():
            assert self._openai_client is not None
            response = await self._openai_client.chat.completions.create(
                model=self._local_model_id(),
                messages=messages,
                **request_kwargs,
            )
            return self._message_to_dict(response.choices[0].message)

        response = await litellm.acompletion(
            messages=messages,
            **request_kwargs,
            **self._common_kwargs(),
        )
        return self._message_to_dict(response.choices[0].message)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> str:
        """Return a single completion from the configured model.

        Args:
            messages: OpenAI-style message list.

        Returns:
            The model's reply as a plain string.

        Raises:
            litellm.exceptions.APIError: On provider-level errors.
        """
        assistant_message = await self.complete_with_message(
            messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return assistant_message.get("content") or ""

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks from a streaming completion.

        Args:
            messages: OpenAI-style message list.

        Yields:
            Text delta strings as they are received from the model.
        """
        request_kwargs: dict[str, Any] = {}
        if tools is not None:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice

        if self._is_local_provider():
            assert self._openai_client is not None
            response = await self._openai_client.chat.completions.create(
                model=self._local_model_id(),
                messages=messages,
                stream=True,
                **request_kwargs,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        response = await litellm.acompletion(
            messages=messages,
            stream=True,
            **request_kwargs,
            **self._common_kwargs(),
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
