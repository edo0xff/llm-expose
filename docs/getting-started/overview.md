# Overview

`llm-expose` is designed for teams that want to expose LLM workflows in chat platforms with a repeatable, scriptable CLI.

Core concepts:

- Model config: where requests are routed (LiteLLM/local/OpenAI-compatible).
- Channel config: how messages are received and sent.
- Pairing: allow-list of chat IDs that can interact with a channel.
- MCP: optional external tools and capabilities attached per channel.

Recommended reading order:

1. [Installation](installation.md)
2. [Quick Start](quick-start.md)
3. [Channel Configuration](../guides/channel-configuration.md)
4. [Provider Configuration](../guides/provider-configuration.md)
5. [MCP Integration](../guides/mcp-integration.md)
6. [Deployment (Docker and Compose)](../guides/deployment.md)
