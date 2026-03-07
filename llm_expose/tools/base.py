"""Abstract base class for tool/function-calling support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Interface for optional LLM function/tool calling support.

    Concrete subclasses represent callable tools that can be exposed to the
    LLM via its function-calling API (e.g. OpenAI's ``tools`` parameter).

    TODO: Implement concrete tool examples (e.g. web search, calculator).
    TODO: Integrate tool execution into the orchestrator message loop.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, snake_case name of the tool (used in the LLM's API call)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON-Schema dict that describes the tool's input parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Invoke the tool with the provided keyword arguments.

        Args:
            **kwargs: Arguments validated against :attr:`parameters_schema`.

        Returns:
            The tool's result as a plain string to be passed back to the LLM.
        """
