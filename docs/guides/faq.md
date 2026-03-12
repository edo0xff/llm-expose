# Frequently Asked Questions

## General Questions

### Q: What is llm-expose?

A: **llm-expose** is a CLI tool that exposes LLM-powered assistants through messaging platforms like Telegram and Discord. It provides a channel-first workflow where you configure providers, attach channels, control access through pairings, and optionally integrate MCP servers for tool-aware completions.

### Q: What models are supported?

A: llm-expose uses **LiteLLM** as its provider, which supports 100+ models from providers including:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini, PaLM)
- Meta (Llama via Together AI)
- Open-source models via local endpoints
- And many more

### Q: Can I use local models?

A: Yes! You can connect to any **OpenAI-compatible endpoint**, which includes:
- Ollama (local LLMs)
- vLLM
- LocalAI
- LM Studio
- LlamaCPP with OpenAI server mode

Configure them as a provider with the endpoint URL.

### Q: Is this production-ready?

A: llm-expose is actively maintained and used in production setups. However, it's still evolving. For production deployments, review the [Deployment Guide](deployment.md) and test thoroughly in your environment.

---

## Installation & Setup

### Q: Should I use interactive mode or headless mode?

A: Use **interactive mode** by default for manual setup. Most commands prompt for missing values automatically.

Use **headless mode** for automation/CI:

- Add `--no-input` to disable prompts and fail fast on missing required values.
- Add `-y` / `--yes` when a command can ask for confirmation (for example overwrite/start flows).

### Q: I got "no matches found: .[dev]" error during dev install. How do I fix it?

A: On macOS/Linux with **zsh**, the glob expansion needs quoting:

```bash
pip install -e '.[dev]'
```

If using bash or other shells, this shouldn't be necessary, but it never hurts to quote it.

### Q: Where is my configuration stored?

A: Configuration is stored in your platform's standard config directory:
- **macOS/Linux**: `~/.config/llm-expose/`
- **Windows**: `%APPDATA%\llm-expose\`

You can manually edit these files or use the CLI commands.

### Q: How do I uninstall or reset everything?

A: To reset your configuration, remove the config directory:

```bash
rm -rf ~/.config/llm-expose/
```

To uninstall the package:

```bash
pip uninstall llm-expose
```

### Q: Can I run multiple instances simultaneously?

A: Yes! Each channel has its own configuration and runtime. You can start multiple channels in separate terminal sessions or use process managers like systemd, supervisord, or Docker containers.

---

## Channel Configuration

### Q: What's the difference between Telegram and Discord channels?

A: Both are messaging platforms with different architectures:

| Feature | Telegram | Discord |
|---------|----------|---------|
| **User ID Type** | Chat ID (numeric) | User ID (numeric) or Server ID |
| **Setup** | Token-only (simpler) | Token + Server ID (more setup) |
| **Scaling** | Good for personal/small teams | Better for communities/servers |
| **Rate Limits** | Generous | Per-endpoint rate limits |

See [Channel Configuration](channel-configuration.md) for detailed setup.

### Q: How do I find my Telegram Chat ID?

A: Multiple ways:

1. **Using a bot URL**: Message `@userinfobot` and it will reply with your Chat ID
2. **Using the API**: Send a message to your bot and check the payload
3. **From the web client**: Check the URL when you open a chat (contains the ID)

### Q: How do I set up a Discord bot?

A: Follow these steps:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create → "New Application"
3. Go to "Bot" → "Add Bot"
4. Copy the **token** (this is your bot token)
5. Under "OAuth2" → "URL Generator", select `bot` and `send_messages` scopes
6. Use the generated URL to invite your bot to your server
7. Configure llm-expose with the token and server ID

### Q: Can I use one model for multiple channels?

A: Yes! Models are global. Once you add a model, all channels can use it. This lets you balance costs or use different models per channel easily.

---

## Pairing & Access Control

### Q: How does the pairing model work?

A: Pairing is how llm-expose controls **who can use the service**:

- When an unpaired user/chat sends a message, they get: `This instance is not paired. Run llm-expose add pair <channel-id>`
- You explicitly add each allowed ID with: `llm-expose add pair <id> --channel <channel-name>`
- Only paired IDs receive responses

This is useful for:
- Private bots (control exactly who uses it)
- Preventing abuse/costs
- Multi-tenant setups

### Q: Can I pair groups/servers vs individual users?

A: Yes! You can pair:
- Individual users (Telegram Chat ID or Discord User ID)
- Group chats (Telegram Group Chat ID or Discord Server/Channel ID)

The service will only serve paired IDs, regardless of type.

### Q: What if I want unrestricted access?

A: Pair a wildcard or all relevant IDs manually. There's no "allow all" setting by design—pairing is explicit for security and cost management.

---

## MCP Integration

### Q: What is MCP and why would I use it?

A: **MCP** (Model Context Protocol) is a standard for connecting tools to LLMs. Integration lets your assistant:
- Call tools (fetch URLs, query APIs, read files)
- Have tool-aware responses
- Execute multi-step workflows

Example: Your Discord assistant can summarize URLs, query databases, or run server commands.

### Q: Which MCP servers are supported?

A: llm-expose supports **any MCP server** that implements the protocol. Popular examples:

- **filesystem**: Read/write files and directories
- **postgres**: Query and modify databases
- **memory**: Persistent conversation memory
- **git**: Repository operations
- **web**: URL fetching and scraping
- **slack**: Interact with Slack workspaces
- **github**: Repository and issue management

See [MCP Integration Guide](mcp-integration.md) for setup details.

### Q: Can I write custom MCP servers?

A: Absolutely! You can create custom MCP servers in Python, JavaScript, or any language. The protocol is language-agnostic. See the [official MCP documentation](https://modelcontextprotocol.io/).

### Q: What happens if an MCP tool fails?

A: The LLM receives the error message and can decide how to handle it. Common behaviors:
- Retry with different parameters
- Inform the user
- Fall back to a different approach

---

## Providers & Models

### Q: How do I add a new model provider?

A: Use the CLI command:

```bash
llm-expose add model --name my-model --provider openai --model-id gpt-4o
```

Supported provider options depend on LiteLLM. See [Provider Configuration](provider-configuration.md).

### Q: Can I use environment variables for API keys?

A: Yes! Set your API keys as environment variables (e.g., `OPENAI_API_KEY`) and llm-expose will discover them automatically via LiteLLM.

### Q: What if my API key is invalid?

A: You'll see an error when starting the channel. Check:

1. The key is set correctly in environment or configuration
2. The provider endpoint is correct
3. The account has available API quota
4. Network connectivity to the provider

See [Troubleshooting](troubleshooting.md) for debug steps.

### Q: How do I switch models without restarting?

A: Models are configured per-channel. To switch:

1. Stop the running instance (`Ctrl+C`)
2. Update the channel config: `llm-expose config set --channel <name> --model-id <new-model>`
3. Restart the channel

You can also maintain multiple channel configs with different models.

---

## Deployment

### Q: Can I deploy llm-expose to the cloud?

A: Yes! Popular options:

- **Docker**: Containerize with Docker (see [Deployment Guide](deployment.md))
- **Heroku/Railway**: Deploy as a worker/service
- **AWS/GCP/Azure**: Run on EC2, Cloud Run, App Service, etc.
- **VPS**: Run on any Linux server (DigitalOcean, Linode, etc.)
- **Raspberry Pi**: Great for local deployments

The key is keeping your API keys secure (use environment variables) and ensuring the server can reach Telegram/Discord APIs.

### Q: How do I keep llm-expose running 24/7?

A: Use a process manager:

- **systemd** (Linux): Native service management
- **supervisord**: Cross-platform process supervision
- **Docker**: Container orchestration (Docker Compose, Kubernetes)
- **PM2** (Node.js): JavaScript process manager
- **systemd-timers** or cron: For scheduled tasks

See [Deployment Guide](deployment.md) for examples.

### Q: What are typical costs?

A: Costs depend on:

- **LLM API**: OpenAI GPT-4o (~$0.01-0.03 per 1K tokens)
- **Infrastructure**: Minimal (cheap VPS or container)
- **Bandwidth**: Usually free

Expected cost for light usage: $10-50/month.

---

## Troubleshooting & Common Issues

### Q: My bot doesn't respond to messages. What should I do?

A: Start with debugging steps:

1. Check the channel is running: `llm-expose start --channel <name>`
2. Verify the user/chat is paired: `llm-expose list pairs --channel <name>`
3. Check logs for errors (usually printed to console)
4. Verify API key is valid

For more details, see the [Troubleshooting Guide](troubleshooting.md).

### Q: How do I check logs?

A: Logs are printed to the terminal by default. For persistent logging, redirect output:

```bash
llm-expose start --channel my-channel > bot.log 2>&1 &
```

For production, use systemd journalctl or your container logs.

### Q: Can I run multiple models in parallel?

A: Each **channel** has one model. To run multiple models, create multiple channels (each with different models) and start them independently.

---

## Contributing & Community

### Q: How can I contribute?

A: We welcome contributions! See [Contributing Guide](../contributing.md) for:

- How to set up the development environment
- Code style and testing requirements
- How to submit PRs
- Reporting bugs and requesting features

### Q: Where can I report bugs?

A: Open an issue on [GitHub Issues](https://github.com/edo0xff/llm-expose/issues). Include:

- What you were trying to do
- Error message/logs
- Python version and OS
- Steps to reproduce

### Q: Is there a community or support channel?

A: Not yet, but you can:

- Open GitHub Discussions for questions
- Report issues on GitHub Issues
- Email or reach out via GitHub profile

---

## Didn't find your question?

If your question isn't here, please:

1. Check the [Troubleshooting Guide](troubleshooting.md) for known issues
2. Review specific guides: [Channel Config](channel-configuration.md), [MCP Integration](mcp-integration.md), [Deployment](deployment.md)
3. Open a [GitHub Discussion](https://github.com/edo0xff/llm-expose/discussions) or [Issue](https://github.com/edo0xff/llm-expose/issues)

We're here to help!
