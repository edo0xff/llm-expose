# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with llm-expose.

## Quick Diagnostics

### Prerequisites Check

Before diving into specific issues, verify your setup:

```bash
# Check Python version (3.11+)
python --version

# Check installation
pip show llm-expose

# Check CLI works
llm-expose --version

# List current configuration
llm-expose list models
llm-expose list channels
llm-expose list pairs
```

### Enable Debug Logging

Most issues are easier to diagnose with detailed logs:

```bash
# Set debug logging level
export LOGLEVEL=DEBUG

# Run your channel with debug output
llm-expose start --channel my-channel
```

---

## Installation Issues

### "Command not found: llm-expose"

**Symptom**: `bash: llm-expose: command not found`

**Possible causes**:
1. Package not installed properly
2. Virtual environment not activated
3. Installation failed silently

**Solutions**:

```bash
# Verify installation
pip show llm-expose

# If not found, reinstall
pip install -e .

# Check it's in your PATH
which llm-expose
python -m llm_expose --version

# If using virtualenv, ensure it's activated
source venv/bin/activate  # or conda activate <env>
```

### "ModuleNotFoundError: No module named 'llm_expose'"

**Symptom**: Error when trying to import or run llm-expose

**Possible causes**:
1. Development installation incomplete
2. Dependencies not installed
3. Wrong Python version

**Solutions**:

```bash
# Reinstall with dev dependencies
pip install -e '.[dev]'

# Check Python version (3.11+ required)
python --version

# Verify all dependencies installed
pip list | grep -i 'discord\|telegram\|litellm'
```

### zsh: "no matches found: .[dev]"

**Symptom**: When running `pip install -e .[dev]`, you get expansion error on zsh

**Cause**: Zsh glob expansion treats `[dev]` as a pattern

**Solution**:

```bash
# Quote the bracket expression
pip install -e '.[dev]'

# Or escape it
pip install -e .\[dev\]
```

---

## Configuration Issues

### "No models found"

**Symptom**: `llm-expose list models` returns empty list

**Possible causes**:
1. No models configured yet
2. Configuration file not found
3. Configuration file corrupted

**Solutions**:

```bash
# Add a test model
llm-expose add model --name test-gpt4 --provider openai --model-id gpt-4o-mini

# Check configuration file location
echo $XDG_CONFIG_HOME  # or ~/.config on Linux/macOS

# Manually inspect config
cat ~/.config/llm-expose/models.json

# Reset configuration if corrupted
rm -rf ~/.config/llm-expose/
llm-expose list models  # Will initialize with empty config
```

### "Channel configuration not found"

**Symptom**: Error when trying to start a channel that should exist

**Possible causes**:
1. Channel name typo
2. Channel configuration file missing
3. Configuration directory moved or lost

**Solutions**:

```bash
# List all channels to verify names
llm-expose list channels

# Check exact spelling (case-sensitive)
llm-expose start --channel "ExactChannelName"

# Recreate the channel
llm-expose add channel  # Interactive setup

# Verify configuration file exists
ls ~/.config/llm-expose/channels/
```

### "Invalid configuration"

**Symptom**: Error message about invalid JSON or configuration format

**Possible cause**: Configuration file corrupted (manual edit, incomplete write, etc.)

**Solutions**:

```bash
# Backup and inspect
cp ~/.config/llm-expose/channels.json ~/.config/llm-expose/channels.json.backup

# Check JSON validity
python -m json.tool ~/.config/llm-expose/channels.json

# If invalid, restore from backup or reconfigure
llm-expose add channel
```

---

## Authentication & API Keys

### "Invalid API key" or "Unauthorized"

**Symptom**: Error when starting channel, won't authenticate with provider

**Possible causes**:
1. API key wrong or expired
2. API key not set in environment
3. API key has insufficient permissions

**Solutions**:

```bash
# Check if API key is set
echo $OPENAI_API_KEY  # For OpenAI, or other provider vars

# Set the API key (Linux/macOS)
export OPENAI_API_KEY="sk-..."

# Test API key authentication
python -c "from litellm import completion; completion(model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'test'}])"

# Check provider documentation for correct env var name
# OpenAI: OPENAI_API_KEY
# Anthropic: ANTHROPIC_API_KEY
# Google: GOOGLE_API_KEY
```

### "API key missing in environment"

**Symptom**: Error indicating API key isn't found

**Solutions**:

```bash
# Set it permanently in your shell config
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc

# Or set it at runtime before starting
export OPENAI_API_KEY="sk-..." && llm-expose start --channel my-channel
```

### "Quota exceeded" or "Rate limited"

**Symptom**: API returns quota or rate limit errors

**Solutions**:

1. **Check your account quota**: Log into your provider dashboard (OpenAI, Anthropic, etc.)
2. **Wait for reset**: Rate limits typically reset hourly or daily
3. **Reduce traffic**: Limit how many users/messages route to that model
4. **Upgrade account**: Increase quota in provider settings
5. **Use a different model**: Switch to a model with different rate limits

---

## Channel Connection Issues

### Telegram Bot Not Responding

**Symptom**: Messages sent to bot don't get responses

#### Step 1: Verify Bot Token

```bash
# Test the token works
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe

# Should return: {"ok":true,"result":{"id":..., ...}}
```

If this fails, your token is invalid.

#### Step 2: Check Channel Configuration

```bash
llm-expose list channels
llm-expose list pairs --channel <channel-name>
```

Verify the channel is configured with the correct token.

#### Step 3: Check If Channel is Running

```bash
# Make sure you're running the channel
llm-expose start --channel my-telegram

# It should print: "Bot running, listening for messages..."
```

#### Step 4: Verify User is Paired

```bash
# Get your Telegram Chat ID
# Send /start to @userinfobot in Telegram - it will reply with your ID

# Pair yourself
llm-expose add pair <YOUR_CHAT_ID> --channel my-telegram

# Verify the pair
llm-expose list pairs --channel my-telegram
```

#### Step 5: Check Logs

```bash
# Run with debug output
export LOGLEVEL=DEBUG
llm-expose start --channel my-telegram

# Look for error messages about webhook, polling, or API calls
```

### Discord Bot Not Responding

**Symptom**: Messages sent to Discord bot get no response

#### Step 1: Verify Bot Token

```bash
# Test the token works
curl -I https://discord.com/api/v10/users/@me \
  -H "Authorization: Bot <YOUR_TOKEN>"

# Should return 200 OK status
```

#### Step 2: Verify Bot Permissions

In Discord Server Settings:

1. Go to "Roles"
2. Select your bot role
3. Ensure these permissions are checked:
   - Send Messages
   - Read Messages/View Channels
   - Embed Links
   - Attach Files

#### Step 3: Check Guild/Channel Configuration

```bash
llm-expose list channels
# Verify the 'server_id' and 'channel_id' match your Discord server
```

#### Step 4: Verify User is Paired

```bash
# Get your Discord User ID (enable Developer Mode, right-click username, Copy User ID)

# Pair yourself
llm-expose add pair <YOUR_USER_ID> --channel my-discord

# Verify
llm-expose list pairs --channel my-discord
```

#### Step 5: Check Intents

If the bot isn't receiving messages, Discord intents might be disabled:

1. In [Discord Developer Portal](https://discord.com/developers), select your app
2. Go to "Bot" section
3. Scroll to "Intents" and enable:
   - ✅ Message Content Intent
   - ✅ Server Members Intent (optional, for prefixes)

#### Step 6: Test with Logs

```bash
export LOGLEVEL=DEBUG
llm-expose start --channel my-discord

# Send a message in Discord and watch the logs
```

---

## LLM Model Issues

### "Model not found" or "Invalid model"

**Symptom**: Error when trying to use a specific model

**Possible causes**:
1. Model name misspelled
2. Model doesn't exist in provider
3. Account doesn't have access to that model

**Solutions**:

```bash
# Verify model is registered
llm-expose list models

# Verify the model ID is correct for the provider
# For OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
# For Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
# For Google: gemini-pro, gemini-pro-vision

# Update a model's name if it's wrong
llm-expose config set --model my-gpt4 --model-id gpt-4o

# Test the model directly
python -c "
from litellm import completion
response = completion(model='gpt-4o', messages=[{'role': 'user', 'content': 'say hi'}])
print(response)
"
```

### "Connection timeout" to Provider

**Symptom**: LLM requests time out or fail with connection errors

**Possible causes**:
1. Network connectivity issue
2. Provider endpoint down
3. Firewall blocking connection
4. Large requests timing out

**Solutions**:

```bash
# Test connectivity to provider
ping api.openai.com  # Example for OpenAI

# Check endpoint is correct
# OpenAI: https://api.openai.com/v1
# Anthropic: https://api.anthropic.com/v1

# Set a custom timeout (if supported by your provider)
export LLM_TIMEOUT=30

# Retry with smaller requests
# Use a faster/smaller model if available
# gpt-4o → gpt-4o-mini for faster responses
```

### "Empty response" or "No completion"

**Symptom**: Model responds but with empty or incomplete messages

**Possible causes**:
1. Model's context window exceeded
2. Provider returning incomplete response
3. Message formatting issue

**Solutions**:

```bash
# Check message is being sent correctly
export LOGLEVEL=DEBUG
llm-expose start --channel my-channel

# Try with a simpler message first
# Try a shorter model that processes faster
llm-expose config set --channel my-channel --model gpt-4o-mini
```

---

## MCP Integration Issues

### "MCP server connection failed"

**Symptom**: Error when trying to connect to MCP server

**Possible causes**:
1. MCP server not running
2. MCP server port/socket incorrect
3. MCP server crashed

**Solutions**:

```bash
# Verify MCP server is running
# For local MCP servers, ensure they're started first:
npx @modelcontextprotocol/server-filesystem /path/to/files

# Check configuration
llm-expose list mcp

# Test MCP server directly
python -c "
import stdio
from mcp.client import Client
client = Client(stdio)
client.connect()
print(client.list_tools())
"
```

### "Tool call failed" or "Tool not available"

**Symptom**: MCP tool exists but fails when called

**Possible causes**:
1. Tool parameters incorrect
2. Tool returned an error
3. Permission issue (e.g., file access)

**Solutions**:

```bash
# Check available tools
# Add debug logging to see tool parameters
export LOGLEVEL=DEBUG

# Verify file permissions if using filesystem tool
# Make sure directories are readable by the llm-expose process
ls -la /path/to/files  # Should show read permissions

# Check MCP server logs for tool errors
```

---

## Performance Issues

### "Bot is slow to respond"

**Symptom**: Long delays between user message and bot response

**Possible causes**:
1. Model is slow (e.g., larger models)
2. Network latency to provider
3. MCP tools are slow
4. System resources limited

**Solutions**:

```bash
# Use a faster model
llm-expose config set --channel my-channel --model gpt-4o-mini

# Check system resources (CPU, RAM, disk)
top        # macOS/Linux
# or
Get-Process  # Windows

# Reduce MCP tool complexity
# Avoid very large file reads or network calls

# Monitor response times
export LOGLEVEL=DEBUG
# Look for timing information in logs
```

### "High memory usage"

**Symptom**: Bot process consuming lots of RAM

**Possible causes**:
1. Long conversations (context window growing)
2. MCP tool handling large files
3. Memory leak in dependencies

**Solutions**:

```bash
# Restart the bot regularly (daily cron job)
0 2 * * * pkill -f "llm-expose start"

# Use a model with smaller context window
# Or implement conversation reset/cleanup

# Monitor memory
watch -n 1 'ps aux | grep llm-expose'
```

---

## Network & Deployment Issues

### "Can't reach Telegram/Discord API"

**Symptom**: Network errors when connecting to messaging platform

**Possible causes**:
1. Firewall blocking outbound connections
2. DNS resolution failing
3. ISP/network blocking

**Solutions**:

```bash
# Test connectivity
curl -v https://api.telegram.org/  # For Telegram
curl -v https://discord.com/api/v10/  # For Discord

# Check DNS
nslookup api.telegram.org

# If corporate firewall: use proxy (if supported by your setup)
# Check company HTTPS proxy settings
```

### Deployed bot stops responding after a while

**Symptom**: Bot works fine initially but becomes unresponsive

**Possible causes**:
1. Process crashed or hung
2. API rate limits
3. Network timeout
4. Memory exhaust

**Solutions**:

```bash
# Use process manager to auto-restart
# systemd example:
[Service]
Type=simple
ExecStart=/usr/bin/python -m llm_expose start --channel my-channel
Restart=always
RestartSec=10s

# Or Docker with restart policy
docker run --restart=always ...

# Regular health checks and logs
# Monitor process and restart if needed (systemd/supervisord)
```

---

## Getting More Help

If you've tried these steps and still have issues:

1. **Check existing issues**: https://github.com/edo0xff/llm-expose/issues
2. **Enable debug logging**: `export LOGLEVEL=DEBUG`
3. **Collect information**:
   ```bash
   python --version
   pip show llm-expose
   pip show litellm discord python-telegram-bot
   ```
4. **Open a GitHub issue** with:
   - What you were trying to do
   - Error messages and logs
   - Configuration (without sensitive keys)
   - Steps to reproduce
5. **Reach out**: Check GitHub profile for contact options

---

## Common Error Messages Reference

| Error | Meaning | Quick Fix |
|-------|---------|-----------|
| "Command not found" | Not installed or not in PATH | `pip install -e .` |
| "Invalid API key" | API key wrong or missing | Check environment variable |
| "No models found" | No models configured | `llm-expose add model` |
| "Connection timeout" | Can't reach provider | Check network/firewall |
| "Unpaired ID" | User not paired | `llm-expose add pair <id>` |
| "Rate limited" | Too many requests | Wait or upgrade account |
| "Permission denied" | File/directory access issue | Check permissions |

---

Have you found an issue not covered here? Please [open a GitHub issue](https://github.com/edo0xff/llm-expose/issues) so we can improve this guide!
