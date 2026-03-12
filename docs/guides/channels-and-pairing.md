# Channels and Pairing

Channels define where your assistant is reachable. Pairing defines who is allowed to interact.

For platform setup details and full channel examples, see [Channel Configuration](channel-configuration.md).

## Pairing Commands

```bash
llm-expose add pair <id> --channel <channel-name>
llm-expose list pairs
llm-expose list pairs --channel <channel-name>
llm-expose delete pair <id> --channel <channel-name>
```

Pairings are scoped by channel configuration, enabling different access control per channel.

## Common pairing workflows

Pair a Telegram chat ID:

```bash
llm-expose add pair 123456789 --channel support-telegram --no-input
```

Pair a Discord channel ID:

```bash
llm-expose add pair 987654321098765432 --channel ops-discord --no-input
```

Inspect current pairings:

```bash
llm-expose list pairs
llm-expose list pairs --channel support-telegram
```

Remove access for one ID:

```bash
llm-expose delete pair 123456789 --channel support-telegram -y --no-input
```

## Pairing behavior

- Pairing is channel-scoped, not global.
- If an incoming ID is not paired, the runtime refuses to reply.
- Add all IDs that should be able to interact (one command per ID).

## Related guides

- [Channel Configuration](channel-configuration.md)
- [MCP Integration](mcp-integration.md)
- [Deployment (Docker and Compose)](deployment.md)
