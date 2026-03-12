# Quick Start

This page gives copy-paste command paths for Telegram and Discord in headless mode.
For platform setup details, troubleshooting, and production examples, continue to the guides:

- [Channel Configuration](../guides/channel-configuration.md)
- [Provider Configuration](../guides/provider-configuration.md)
- [MCP Integration](../guides/mcp-integration.md)
- [Deployment (Docker and Compose)](../guides/deployment.md)

## Telegram in 4 steps

1. Add a model:

```bash
llm-expose add model \
	--name gpt4o-mini \
	--provider openai \
	--model-id gpt-4o-mini \
	-y --no-input
```

2. Add a Telegram channel:

```bash
llm-expose add channel \
	--name team-telegram \
	--client-type telegram \
	--bot-token "123456789:AAExampleTelegramToken" \
	--model-name gpt4o-mini \
	-y --no-input
```

3. Pair the allowed Telegram chat ID:

```bash
llm-expose add pair 123456789 --channel team-telegram --no-input
```

4. Start the runtime:

```bash
llm-expose start --channel team-telegram -y --no-input
```

## Discord in 4 steps

1. Reuse the same model or create a dedicated one:

```bash
llm-expose add model \
	--name gpt4o-discord \
	--provider openai \
	--model-id gpt-4o-mini \
	-y --no-input
```

2. Add a Discord channel config:

```bash
llm-expose add channel \
	--name ops-discord \
	--client-type discord \
	--bot-token "YOUR_DISCORD_BOT_TOKEN" \
	--model-name gpt4o-discord \
	-y --no-input
```

3. Pair the allowed Discord channel ID:

```bash
llm-expose add pair 987654321098765432 --channel ops-discord --no-input
```

4. Start the runtime:

```bash
llm-expose start --channel ops-discord -y --no-input
```

## Add MCP servers (optional)

1. Add an MCP server definition:

```bash
llm-expose add mcp \
	--name web-search \
	--transport stdio \
	--command uvx \
	--args mcp-server-web-search \
	--tool-confirmation never \
	-y --no-input
```

2. Attach it to a channel:

```bash
llm-expose add channel \
	--name team-telegram \
	--client-type telegram \
	--bot-token "123456789:AAExampleTelegramToken" \
	--model-name gpt4o-mini \
	--mcp-server web-search \
	-y --no-input
```

See [MCP Integration](../guides/mcp-integration.md) for transport-specific server configs and approval policies.

## Verify configuration

```bash
llm-expose list models
llm-expose list channels
llm-expose list pairs
llm-expose list mcp
```

Use `llm-expose <command> --help` and `llm-expose <command> <subcommand> --help` for full options.
