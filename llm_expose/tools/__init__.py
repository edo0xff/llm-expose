"""Tools package for llm-expose.

This package provides the base interface for optional function/tool calling
support that can be exposed to the LLM.

TODO: Add concrete tool implementations (web search, calculator, etc.).
"""

from llm_expose.tools.base import BaseTool

__all__ = ["BaseTool"]
