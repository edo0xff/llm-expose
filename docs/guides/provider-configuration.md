# Provider Configuration

This guide covers practical model/provider setup patterns for online APIs and local OpenAI-compatible runtimes.

`llm-expose add model` is interactive by default. For manual setup, run it without flags and select provider/model through prompts.
Use the explicit examples below when you want repeatable headless commands (`--no-input`, and `-y` for overwrite confirmation).

## Provider model

Each saved model includes:

- `provider_name`: provider routing key (for example `openai`, `anthropic`, or `local`)
- `model`: provider model identifier
- `api_key`: optional key (prefer environment variables in production)
- `base_url`: optional custom endpoint (common for local/self-hosted gateways)

## OpenAI example

Set credentials:

```bash
export OPENAI_API_KEY="sk-..."
```

Create model config:

```bash
llm-expose add model \
  --name gpt4o-mini \
  --provider openai \
  --model-id gpt-4o-mini \
  -y --no-input
```

## Anthropic example

Set credentials:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Create model config:

```bash
llm-expose add model \
  --name claude-sonnet \
  --provider anthropic \
  --model-id claude-3-5-sonnet-latest \
  -y --no-input
```

## Local OpenAI-compatible example

Use this for local gateways such as LM Studio, vLLM, or OpenAI-compatible proxies.

```bash
llm-expose add model \
  --name local-llm \
  --provider local \
  --model-id mistral-7b-instruct \
  --base-url http://localhost:1234/v1 \
  -y --no-input
```

Notes:

- `--base-url` defaults to `http://localhost:1234/v1` for `--provider local` if omitted.
- Some local gateways require an API key header even when value is not validated.

## Use provider configs on channels

Bind a saved model to channel config with `--model-name`:

```bash
llm-expose add channel \
  --name support-telegram \
  --client-type telegram \
  --bot-token "123456789:AAExampleTelegramToken" \
  --model-name gpt4o-mini \
  -y --no-input
```

Switch model routing by updating channel with a different model name:

```bash
llm-expose add channel \
  --name support-telegram \
  --client-type telegram \
  --bot-token "123456789:AAExampleTelegramToken" \
  --model-name claude-sonnet \
  -y --no-input
```

## Example model YAML

OpenAI:

```yaml
provider_name: openai
model: gpt-4o-mini
api_key: null
base_url: null
supports_vision: null
```

Anthropic:

```yaml
provider_name: anthropic
model: claude-3-5-sonnet-latest
api_key: null
base_url: null
supports_vision: null
```

Local:

```yaml
provider_name: local
model: mistral-7b-instruct
api_key: null
base_url: http://localhost:1234/v1
supports_vision: false
```

## Verify model setup

```bash
llm-expose list models
llm-expose list channels
```

## Troubleshooting

- Model not found on startup: check `--model-name` matches an existing saved model name.
- Auth failures with online providers: verify exported API key variable for that provider.
- Local connection failures: verify `base_url` and local server reachability.
- Wrong provider behavior: confirm `provider_name` and `model` values in saved config.