"""Tools package for llm-expose.

The runtime MCP path is provider-managed via LiteLLM (server-side tool
discovery/execution). ``BaseTool`` remains available as a legacy local-tool
extension point.
"""

from llm_expose.tools.base import BaseTool

__all__ = ["BaseTool"]
