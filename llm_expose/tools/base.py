"""Abstract base class for tool/function-calling support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Legacy interface for optional local tool execution.

    Concrete subclasses represent callable tools executed inside this
    application. The current production path uses LiteLLM MCP tools directly,
    so this interface is kept primarily for backward compatibility and future
    local-only integrations.

    TODO: Remove once local tool execution is either fully implemented or
        formally deprecated across public API/docs.
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
