# llm-expose
CLI tool for expose LLM's using common chat services (like telegram).

## Pairing system

Incoming chat/channel IDs must be paired before this service replies.

When an unpaired chat sends a message, the service replies with:

`This instance is not paired. Run llm-expose add pair <channel-id>`

Pairings are scoped per saved channel configuration and persisted server-side.

### Commands

- `llm-expose add pair <id> --channel <channel-name>`
- `llm-expose list pairs`
- `llm-expose list pairs --channel <channel-name>`
- `llm-expose delete pair <id> --channel <channel-name>`
