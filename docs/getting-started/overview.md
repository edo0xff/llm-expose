# Overview

`llm-expose` is designed for teams that want to expose LLM workflows in chat platforms with a repeatable, scriptable CLI.

Core concepts:

- Model config: where requests are routed (LiteLLM/local/OpenAI-compatible).
- Channel config: how messages are received and sent.
- Pairing: allow-list of chat IDs that can interact with a channel.
- MCP: optional external tools and capabilities attached per channel.
