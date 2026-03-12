# Channels and Pairing

Channels define where your assistant is reachable. Pairing defines who is allowed to interact.

## Pairing Commands

```bash
llm-expose add pair <id> --channel <channel-name>
llm-expose list pairs
llm-expose list pairs --channel <channel-name>
llm-expose delete pair <id> --channel <channel-name>
```

Pairings are scoped by channel configuration, enabling different access control per channel.
