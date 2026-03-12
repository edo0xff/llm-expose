# Channel Configuration

This guide shows practical Telegram and Discord setups with repeatable, headless CLI commands.

## Telegram setup

### Step 1: Create a bot token

1. Open Telegram and start a chat with `@BotFather`.
2. Run `/newbot` and follow the prompts.
3. Copy the bot token (`123456789:AA...`).

### Step 2: Add or reuse a model

```bash
llm-expose add model \
  --name gpt4o-mini \
  --provider openai \
  --model-id gpt-4o-mini \
  -y --no-input
```

### Step 3: Add the Telegram channel config

```bash
llm-expose add channel \
  --name support-telegram \
  --client-type telegram \
  --bot-token "123456789:AAExampleTelegramToken" \
  --model-name gpt4o-mini \
  --system-prompt-path ./prompts/support.txt \
  -y --no-input
```

### Step 4: Pair allowed chat IDs

```bash
llm-expose add pair 123456789 --channel support-telegram --no-input
```

Use one `add pair` per allowed user/group chat ID.

## Discord setup

### Step 1: Create a bot token

1. Open Discord Developer Portal.
2. Create or select an application, then create a bot.
3. Enable Message Content Intent in Bot -> Privileged Gateway Intents.
4. Copy the bot token.

### Step 2: Add the Discord channel config

```bash
llm-expose add channel \
  --name ops-discord \
  --client-type discord \
  --bot-token "YOUR_DISCORD_BOT_TOKEN" \
  --model-name gpt4o-mini \
  --system-prompt-path ./prompts/ops.txt \
  -y --no-input
```

### Step 3: Pair allowed channel IDs

```bash
llm-expose add pair 987654321098765432 --channel ops-discord --no-input
```

Use one `add pair` per Discord channel ID that should receive replies.

## Attach MCP servers to a channel

Attach MCP servers at channel creation time by repeating `--mcp-server`:

```bash
llm-expose add channel \
  --name ops-discord \
  --client-type discord \
  --bot-token "YOUR_DISCORD_BOT_TOKEN" \
  --model-name gpt4o-mini \
  --mcp-server web-search \
  -y --no-input
```

If an MCP server name does not exist, headless mode fails with an explicit error.

## Example channel YAMLs

Telegram channel config:

```yaml
client_type: telegram
bot_token: "123456789:AAExampleTelegramToken"
mcp_servers:
  - web-search
system_prompt_path: ./prompts/support.txt
model_name: gpt4o-mini
```

Discord channel config:

```yaml
client_type: discord
bot_token: "YOUR_DISCORD_BOT_TOKEN"
mcp_servers:
  - web-search
system_prompt_path: ./prompts/ops.txt
model_name: gpt4o-mini
```

## Prompt file strategy

- Keep prompt files in a dedicated directory like `./prompts/`.
- Use one prompt per channel role (`support.txt`, `ops.txt`, `oncall.txt`).
- If `system_prompt_path` is missing or unreadable at startup, runtime falls back to the default system prompt.

## Start and verify

```bash
llm-expose start --channel support-telegram -y --no-input
# or
llm-expose start --channel ops-discord -y --no-input
```

Verify persisted configuration:

```bash
llm-expose list channels
llm-expose list pairs
llm-expose list models
```

## Troubleshooting

- Telegram reports another polling process: stop any other process using the same bot token.
- Discord bot receives empty content: confirm Message Content Intent is enabled.
- Runtime says channel is not paired: add pair IDs for the exact chat/channel where messages arrive.
- Model is not used: confirm `model_name` points to an existing model config.
