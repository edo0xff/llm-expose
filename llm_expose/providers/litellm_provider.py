"""LiteLLM-backed provider implementation."""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, AsyncIterator

import litellm
from openai import AsyncOpenAI

from llm_expose.config.models import ProviderConfig
from llm_expose.core.content_parts import messages_have_images, strip_image_parts
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
        self._supports_vision = self._detect_vision_support()
        self._last_usage: dict[str, Any] | None = None
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

    def _detect_vision_support(self) -> bool:
        """Determine vision capability from config override or LiteLLM metadata."""
        if self._config.supports_vision is not None:
            return bool(self._config.supports_vision)

        try:
            model_info = litellm.get_model_info(self._config.model.strip())
            if isinstance(model_info, dict):
                supports_vision = model_info.get("supports_vision")
                if isinstance(supports_vision, bool):
                    return supports_vision
        except Exception:
            # Conservative fallback when model metadata cannot be resolved.
            pass

        return False

    def supports_vision(self) -> bool:
        """Return whether the configured model supports image input."""
        return self._supports_vision

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Normalize message payload according to model capabilities."""
        if self._supports_vision or not messages_have_images(messages):
            return messages

        normalized, stripped_count = strip_image_parts(messages)
        if stripped_count:
            warning_message = (
                f"Model '{self._config.model}' does not support vision; "
                f"skipping {stripped_count} image part(s)."
            )
            warnings.warn(warning_message)
        return normalized

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

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_completion_usage(self, response: Any, *, elapsed_ms: int) -> dict[str, Any] | None:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None and isinstance(response, dict):
            usage_obj = response.get("usage")

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        if usage_obj is not None:
            if isinstance(usage_obj, dict):
                prompt_tokens = self._as_int(usage_obj.get("prompt_tokens"))
                completion_tokens = self._as_int(usage_obj.get("completion_tokens"))
                total_tokens = self._as_int(usage_obj.get("total_tokens"))
            else:
                prompt_tokens = self._as_int(getattr(usage_obj, "prompt_tokens", None))
                completion_tokens = self._as_int(getattr(usage_obj, "completion_tokens", None))
                total_tokens = self._as_int(getattr(usage_obj, "total_tokens", None))

        # Compute total when provider omits it but gives partial counters.
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        cost_usd: float | None = None
        try:
            # Estimated value from LiteLLM pricing metadata when available.
            cost_usd = self._as_float(litellm.completion_cost(completion_response=response))
        except Exception:
            cost_usd = None

        if prompt_tokens is None and completion_tokens is None and total_tokens is None and cost_usd is None:
            return None

        model_name = getattr(response, "model", None)
        if isinstance(response, dict):
            model_name = model_name or response.get("model")

        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "model": str(model_name or self._config.model.strip()),
            "latency_ms": int(elapsed_ms),
        }
        return usage

    def get_last_usage(self) -> dict[str, Any] | None:
        """Return the most recent completion usage payload when available."""
        if self._last_usage is None:
            return None
        return dict(self._last_usage)

    async def complete_with_message(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Message:
        """Return the full assistant message payload for tool-call handling."""
        messages = self._prepare_messages(messages)
        self._last_usage = None
        started = time.monotonic()
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
            elapsed_ms = (time.monotonic() - started) * 1000
            self._last_usage = self._extract_completion_usage(response, elapsed_ms=int(elapsed_ms))
            return self._message_to_dict(response.choices[0].message)

        response = await litellm.acompletion(
            messages=messages,
            **request_kwargs,
            **self._common_kwargs(),
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        self._last_usage = self._extract_completion_usage(response, elapsed_ms=int(elapsed_ms))
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
        messages = self._prepare_messages(messages)
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
