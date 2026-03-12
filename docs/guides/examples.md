# Examples & Cookbook

Real-world examples and code snippets to help you get started with llm-expose.

## Basic Setup Examples

### Example 1: Simple Telegram Bot with GPT-4o

Complete setup for a basic Telegram bot powered by GPT-4o.

=== "Step-by-step"

    ```bash
    # 1. Get your Telegram bot token from @BotFather
    # (Create a new bot and copy the token)
    
    # 2. Add the model
    llm-expose add model \
      --name gpt4o \
      --provider openai \
      --model-id gpt-4o
    
    # 3. Create the Telegram channel
    llm-expose add channel
    # When prompted:
    # - Name: telegram-main
    # - Type: telegram
    # - Token: <paste your token>
    
    # 4. Pair your Telegram Chat ID
    # Get your ID from @userinfobot
    llm-expose add pair 123456789 --channel telegram-main
    
    # 5. Start the bot!
    llm-expose start --channel telegram-main
    ```

=== "All at once (with env var)"

    ```bash
    # Set API key
    export OPENAI_API_KEY="sk-..."
    export TELEGRAM_BOT_TOKEN="123:ABC..."
    
    # Add model
    llm-expose add model --name gpt4o --provider openai --model-id gpt-4o
    
    # Create channel with token from env
    llm-expose add channel  # Select Telegram, paste token
    
    # Pair user (replace with your ID)
    llm-expose add pair 123456789 --channel telegram-main
    
    # Run!
    llm-expose start --channel telegram-main
    ```

### Example 2: Discord Bot with Claude 3 Haiku

Budget-friendly Discord bot using smaller Claude model.

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Add model (cheaper, fast)
llm-expose add model \
  --name claude-haiku \
  --provider anthropic \
  --model-id claude-3-haiku-20240307

# Create Discord channel
# When prompted with add channel:
# - Name: discord-support
# - Type: discord
# - Token: <your bot token>
# - Server ID: <your server ID>

# Pair users (Discord User IDs)
llm-expose add pair 987654321 --channel discord-support
llm-expose add pair 567890123 --channel discord-support

# Start
llm-expose start --channel discord-support
```

---

## Advanced Examples

### Example 3: Telegram Bot with File System Access (MCP)

Telegram bot that can read and write files on your system.

=== "Setup"

    ```bash
    # 1. Set API key
    export OPENAI_API_KEY="sk-..."
    export TELEGRAM_BOT_TOKEN="123:ABC..."
    
    # 2. Add model
    llm-expose add model --name gpt4o --provider openai --model-id gpt-4o
    
    # 3. Add MCP server for file system
    llm-expose add mcp
    # When prompted:
    # - Name: filesystem
    # - Type: stdio
    # - Command: npx @modelcontextprotocol/server-filesystem /home/user/shared
    
    # 4. Create Telegram channel
    llm-expose add channel
    
    # 5. Attach MCP server to channel
    llm-expose config set --channel telegram-main --mcp filesystem
    
    # 6. Start!
    llm-expose start --channel telegram-main
    ```

=== "Usage in Telegram"

    ```
    User: Create a summary file for today
    Bot: I'll create a summary file for you...
    <bot creates summary.md in shared folder>
    Bot: Done! I've created summary.md
    
    User: What files are in the shared folder?
    Bot: The files in /home/user/shared are:
    - summary.md (created today)
    - notes.txt
    - data.json
    ```

### Example 4: Discord Community Server with Role-Based Access

Multiple channels for different roles/departments.

```bash
# Setup models
llm-expose add model --name gpt4o-turbo \
  --provider openai \
  --model-id gpt-4o

llm-expose add model --name gpt4o-mini \
  --provider openai \
  --model-id gpt-4o-mini

# Create channels
llm-expose add channel
# • discord-general (gpt4o-mini for cost)

llm-expose add channel
# • discord-support (gpt4o-turbo for better responses)

# Pair support team users
llm-expose add pair 111111111 --channel discord-support  # User 1
llm-expose add pair 222222222 --channel discord-support  # User 2

# Pair all users for general (pair a group chat instead)
llm-expose add pair 333333333 --channel discord-general  # Group/Channel ID

# Start both channels
llm-expose start --channel discord-general &
llm-expose start --channel discord-support
```

### Example 5: Multi-Model Telegram Bot

Different models for different use cases.

```bash
# Add multiple models
llm-expose add model --name fast --provider openai --model-id gpt-4o-mini
llm-expose add model --name smart --provider openai --model-id gpt-4o
llm-expose add model --name creative --provider anthropic --model-id claude-3-opus

# Create separate channels for each
llm-expose add channel  # -> telegram-fast (model: fast)
llm-expose add channel  # -> telegram-smart (model: smart)
llm-expose add channel  # -> telegram-creative (model: creative)

# Pair same user to all channels
llm-expose add pair 123456789 --channel telegram-fast
llm-expose add pair 123456789 --channel telegram-smart
llm-expose add pair 123456789 --channel telegram-creative

# Start all three
for channel in telegram-fast telegram-smart telegram-creative; do
  llm-expose start --channel $channel &
done
```

---

## Integration Examples

### Example 6: Local Development Setup with Ollama

Use a local LLM for free development.

=== "Install Ollama"

    ```bash
    # Download Ollama from https://ollama.ai
    # Or on Linux:
    curl https://ollama.ai/install.sh | sh
    
    # Pull a model
    ollama pull llama2:7b
    # or
    ollama pull mistral
    
    # Start Ollama (it runs on localhost:11434)
    ollama serve
    ```

=== "Configure llm-expose"

    ```bash
    # Add Ollama model (OpenAI-compatible endpoint)
    llm-expose add model \
      --name llama2-local \
      --provider openai \
      --model-id llama2 \
      --api-base http://localhost:11434/v1
    
    # Create channel
    llm-expose add channel
    # (Select Telegram or Discord)
    
    # Start (no API key needed!)
    llm-expose start --channel my-channel
    ```

### Example 7: GitHub Issue Resolver Bot

Discord bot that monitors and comments on GitHub issues (with MCP GitHub).

```bash
# Setup
export OPENAI_API_KEY="sk-..."
export GITHUB_TOKEN="ghp_..."

# Add model
llm-expose add model --name gpt4o --provider openai --model-id gpt-4o

# Add GitHub MCP server
llm-expose add mcp
# • Name: github
# • Command: npx @modelcontextprotocol/server-github

# Create Discord channel for GitHub updates
llm-expose add channel
# • discord-github
# • Model: gpt4o
# • MCP: github

# Pair GitHub bot user/webhook channel
llm-expose add pair <your-github-bot-channel-id> --channel discord-github

# Now you can:
# - Forward GitHub issues to Discord
# - Use bot to analyze and comment on issues
# - Automated triage with the bot's insights
```

---

## Configuration Management

### Example 8: Backup and Version Control Configuration

Keep your configuration in git (securely).

```bash
# Create a config backup
mkdir -p ~/backups
cp -r ~/.config/llm-expose ~/backups/llm-expose-backup

# Create a git repo for configuration (without secrets!)
cd ~/projects
git init llm-expose-config
cd llm-expose-config

# Copy config structure (we'll remove secrets next)
cp -r ~/.config/llm-expose . 

# Create .gitignore to exclude secrets
cat > .gitignore << 'EOF'
# Don't commit API keys!
*.json
!models-schema.json
!channels-schema.json
.env
.env.local
EOF

# For deployment, store secrets in environment variables
cat > .env.example << 'EOF'
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=123:ABC...
ANTHROPIC_API_KEY=sk-ant-...
EOF

# Now safe to commit
git add .gitignore .env.example
git commit -m "Initial config structure"
```

### Example 9: Production Deployment with systemd

Run bots as system services that auto-restart.

=== "Create service file"

    ```bash
    # Create /etc/systemd/system/llm-expose-telegram.service
    sudo tee /etc/systemd/system/llm-expose-telegram.service > /dev/null << 'EOF'
    [Unit]
    Description=LLM Expose Telegram Bot
    After=network.target
    
    [Service]
    Type=simple
    User=llm-user
    Group=llm-user
    WorkingDirectory=/home/llm-user
    Environment="OPENAI_API_KEY=sk-..."
    Environment="TELEGRAM_BOT_TOKEN=123:ABC..."
    ExecStart=/usr/local/bin/llm-expose start --channel telegram-main
    Restart=always
    RestartSec=10
    StandardOutput=journal
    StandardError=journal
    
    [Install]
    WantedBy=multi-user.target
    EOF
    ```

=== "Enable and start"

    ```bash
    # Enable service (starts on boot)
    sudo systemctl enable llm-expose-telegram
    
    # Start service
    sudo systemctl start llm-expose-telegram
    
    # Check status
    sudo systemctl status llm-expose-telegram
    
    # View logs
    sudo journalctl -u llm-expose-telegram -f
    ```

---

## Specialized Use Cases

### Example 10: Personal Knowledge Assistant

Telegram bot with access to your personal wiki/knowledge base.

```bash
# Add model
llm-expose add model --name gpt4o --provider openai --model-id gpt-4o

# Add filesystem MCP for your knowledge base
llm-expose add mcp
# • filesystem accessing ~/documents/wiki/

# Create Telegram channel with MCP
llm-expose add channel
# • telegram-assistant
# • Model: gpt4o
# • MCP: filesystem

# Only pair yourself
llm-expose add pair <your-id> --channel telegram-assistant

# Usage: "Summarize my notes on Python" → Bot reads your wiki files and responds
```

### Example 11: Content Creator Assistant

Discord server for content team with tool access.

```bash
# Models for different tasks
llm-expose add model --name creative --provider anthropic --model-id claude-3-opus
llm-expose add model --name editor --provider openai --model-id gpt-4o

# Discord channels
llm-expose add channel  # discord-writers (uses creative)
llm-expose add channel  # discord-editors (uses editor)

# Pair team members
llm-expose add pair <writer-id> --channel discord-writers
llm-expose add pair <editor-id> --channel discord-editors
llm-expose add pair <manager-id> --channel discord-writers  # Managers see both
llm-expose add pair <manager-id> --channel discord-editors

# With MCP tools, writers can:
# - Ask bot for writing suggestions
# - Get editing feedback
# - Research topics
```

---

## Troubleshooting Common Setup Issues

See the [Troubleshooting Guide](troubleshooting.md) for detailed help with:

- Bot not responding
- API key errors
- Configuration issues
- Performance problems
- Deployment issues

---

## Tips & Tricks

!!! tip "Pro Tips"

    === "Monitoring Multiple Bots"
        ```bash
        # Start all channels at once (in background)
        for channel in telegram-main discord-support; do
          llm-expose start --channel $channel > logs/$channel.log 2>&1 &
        done
        
        # View all logs
        tail -f logs/*.log
        ```

    === "Quick Configuration Reset"
        ```bash
        # To reset without losing everything, backup first:
        cp -r ~/.config/llm-expose ~/.config/llm-expose.backup
        
        # Then carefully update just what you need:
        llm-expose config set --channel mychannel --model new-model-id
        ```

    === "Testing Bot Locally"
        ```bash
        # Use Ollama or local endpoint for free testing
        llm-expose add model --name test --provider openai \
          --model-id llama2 \
          --api-base http://localhost:11434/v1
        
        # No API key needed!
        ```

    === "Keeping Logs for Debugging"
        ```bash
        # Start with persistent logging
        llm-expose start --channel my-channel \
          > ~/.logs/llm-expose-$(date +%Y%m%d).log 2>&1 &
        
        # Check with:
        tail -f ~/.logs/llm-expose-*.log
        ```

---

## Next Steps

- **Explore MCP**: Check the [MCP Integration Guide](mcp-integration.md) for more tool options
- **Deploy to production**: See [Deployment Guide](deployment.md) for cloud options
- **Advanced configuration**: Review [Provider Configuration](provider-configuration.md) for more models
- **Custom development**: Build your own MCP servers or llm-expose extensions

---

Have an example you'd like to share? [Contribute to the docs!](../contributing.md)
