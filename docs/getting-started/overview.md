# Overview

`llm-expose` is designed for teams that want to expose LLM workflows in chat platforms with a repeatable, scriptable CLI.

Core concepts:

- Model config: where requests are routed (LiteLLM/local/OpenAI-compatible).
- Channel config: how messages are received and sent.
- Pairing: allow-list of chat IDs that can interact with a channel.
- MCP: optional external tools and capabilities attached per channel.

## Interactive vs headless mode

`llm-expose` commands are interactive by default. If you omit required flags, the CLI will ask you for the missing values.

Use interactive mode when you are setting up channels manually and want guided prompts.

Use headless mode for CI/CD and scripts:

- `--no-input`: fail immediately if required input is missing.
- `-y` / `--yes`: skip overwrite/start confirmations when needed.

Recommended reading order:

1. [Installation](installation.md)
2. [Quick Start](quick-start.md)
3. [Channel Configuration](../guides/channel-configuration.md)
4. [Provider Configuration](../guides/provider-configuration.md)
5. [MCP Integration](../guides/mcp-integration.md)
6. [Deployment (Docker and Compose)](../guides/deployment.md)
