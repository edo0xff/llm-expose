<div align="center">

<img width="550" alt="LLM Expose Logo" src="docs/assets/logo.png">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Language: Python](https://img.shields.io/badge/Language-Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/edo0xff/llm-expose/test.yml?label=Tests&logo=github)](https://github.com/edo0xff/llm-expose/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/Docs-Live-brightgreen?logo=readthedocs)](https://edo0xff.github.io/llm-expose)

**Expose LLM-powered assistants through messaging platforms such as Telegram and Discord**

🤖 ← 🐈 ← 🌐 ← 📱 ← 🧙‍♂️
</div>

---

`llm-expose` gives you a channel-first CLI workflow: configure providers, attach channels, control pairings, and optionally integrate MCP servers for tool-aware completions.

## Features

- Multi-channel support (Telegram and Discord).
- LiteLLM provider support for broad model compatibility.
- Local OpenAI-compatible endpoint support.
- MCP server integration for tool-aware responses.
- Pairing-based access control per channel.
- CLI-first setup and operations.

## Installation

### Quick Install (One-Liner)

**Linux & macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install.sh | bash
```

**Windows (PowerShell as Administrator):**
```powershell
powershell -ExecutionPolicy Bypass -Command "iex (New-Object Net.WebClient).DownloadString('https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install-windows.ps1')"
```

### From PyPI

```bash
pip install llm-expose
```

### From source

```bash
git clone https://github.com/edo0xff/llm-expose.git
cd llm-expose
pip install -e .
```

### Development install

```bash
pip install -e '.[dev]'
```

> See [scripts/README.md](scripts/README.md) for detailed installation instructions and troubleshooting.

## Quick Start

`llm-expose` is interactive by default, which is usually the fastest path for humans.
Use `--no-input` for headless automation and add `-y` when the command can require confirmation.

1. Configure a model:

```bash
llm-expose add model
```

2. Configure a channel (interactive):

```bash
llm-expose add channel
```

3. Pair an allowed user/chat ID:

```bash
llm-expose add pair 123456789 --channel my-telegram
```

4. Start the channel runtime:

```bash
llm-expose start
```

Headless equivalent (CI/scripts):

```bash
llm-expose add model --name gpt4o-mini --provider openai --model-id gpt-4o-mini -y --no-input
llm-expose add channel --name my-telegram --client-type telegram --bot-token "123456789:AAExampleTelegramToken" --model-name gpt4o-mini -y --no-input
llm-expose add pair 123456789 --channel my-telegram --no-input
llm-expose start --channel my-telegram -y --no-input
```

If you are unsure about available options, run:

```bash
llm-expose --help
llm-expose add --help
llm-expose start --help
```

## Pairing Model

Incoming chat/channel IDs must be explicitly paired before the service replies.

When an unpaired ID sends a message, the service returns:

`This instance is not paired. Run llm-expose add pair <channel-id>`

Pairings are stored per channel configuration.

Common pairing commands:

- `llm-expose add pair <id> --channel <channel-name>`
- `llm-expose list pairs`
- `llm-expose list pairs --channel <channel-name>`
- `llm-expose delete pair <id> --channel <channel-name>`

## Configuration Workflow

`llm-expose` currently uses CLI commands to persist configuration (models, channels, and MCP settings).

Recommended setup order:

1. Add one or more models (`llm-expose add model ...`).
2. Add one or more channels (`llm-expose add channel ...`).
3. Add optional MCP servers (`llm-expose add mcp ...`).
4. Pair allowed IDs (`llm-expose add pair ...`).
5. Run exposure service (`llm-expose start ...`).

## Development

Run quality checks:

```bash
ruff check .
black --check .
mypy llm_expose
pytest
```

## Roadmap

- PyPI release automation.
- Hosted docs site with architecture and API references.
- More channel adapters and provider presets.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
