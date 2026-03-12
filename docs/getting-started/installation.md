# Installation

## Requirements

- Python 3.11 or newer
- Access to at least one provider or local OpenAI-compatible endpoint

## Install from source

```bash
git clone https://github.com/edo0xff/llm-expose.git
cd llm-expose
pip install -e .
```

## Install with development tools

```bash
pip install -e .[dev]
```

In zsh, quote extras to avoid shell globbing:

```bash
pip install -e '.[dev]'
```

## Configure environment variables

Set provider keys as environment variables when possible.

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Optionally override where `llm-expose` stores config:

```bash
export LLM_EXPOSE_CONFIG_DIR="$HOME/.llm-expose"
```

Default layout under the config directory:

- `models/` for saved model definitions
- `channels/` for saved channel definitions
- `pairs.yaml` for pairing allow-lists
- `mcp_servers.yaml` for MCP server and settings config

Keep bot tokens and API keys out of version control and prefer environment-variable injection in deployment.

## Validate CLI

```bash
llm-expose --help
```

Next: [Quick Start](quick-start.md)
