# Deployment (Docker and Compose)

This guide focuses on running `llm-expose` as a long-lived containerized service.

Deployment commands should be headless. Use `--no-input` and `-y` in runtime/start commands so containers never wait for interactive prompts.

For first-time local setup of models/channels/pairs, you can use interactive mode before containerizing:

```bash
llm-expose add model
llm-expose add channel
llm-expose add pair
```

## Deployment model

- One channel runtime per process (`llm-expose start --channel <name>`).
- Config and prompts are mounted into the container.
- Provider secrets are injected through environment variables.
- Container restart policy keeps the service alive.

## Directory layout

Example host layout:

```text
deploy/
  compose.yaml
  .env
  config/
    models/
    channels/
    mcp_servers.yaml
    pairs.yaml
  prompts/
    support.txt
    ops.txt
```

## Dockerfile

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir llm-expose

WORKDIR /app
ENV LLM_EXPOSE_CONFIG_DIR=/config

CMD ["llm-expose", "start", "--channel", "support-telegram", "-y", "--no-input"]
```

## Docker run

```bash
docker build -t llm-expose:latest .

docker run -d \
  --name llm-expose-support \
  --restart unless-stopped \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e LLM_EXPOSE_CONFIG_DIR=/config \
  -v "$PWD/deploy/config:/config" \
  -v "$PWD/deploy/prompts:/prompts:ro" \
  llm-expose:latest
```

## Docker Compose

`compose.yaml`:

```yaml
services:
  llm-expose:
    image: llm-expose:latest
    container_name: llm-expose-support
    restart: unless-stopped
    environment:
      LLM_EXPOSE_CONFIG_DIR: /config
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    command: ["llm-expose", "start", "--channel", "support-telegram", "-y", "--no-input"]
    volumes:
      - ./config:/config
      - ./prompts:/prompts:ro
```

`.env`:

```bash
OPENAI_API_KEY=sk-...
```

Run:

```bash
docker compose up -d
docker compose logs -f llm-expose
```

## Multiple channels

Use one service per channel runtime:

```yaml
services:
  support-telegram:
    image: llm-expose:latest
    restart: unless-stopped
    environment:
      LLM_EXPOSE_CONFIG_DIR: /config
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    command: ["llm-expose", "start", "--channel", "support-telegram", "-y", "--no-input"]
    volumes:
      - ./config:/config
      - ./prompts:/prompts:ro

  ops-discord:
    image: llm-expose:latest
    restart: unless-stopped
    environment:
      LLM_EXPOSE_CONFIG_DIR: /config
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    command: ["llm-expose", "start", "--channel", "ops-discord", "-y", "--no-input"]
    volumes:
      - ./config:/config
      - ./prompts:/prompts:ro
```

## Operational checklist

- Confirm config files exist before startup (`models`, `channels`, `pairs.yaml`, `mcp_servers.yaml`).
- Keep prompt mounts read-only where possible.
- Keep secrets out of config files and inject via environment variables.
- Run `llm-expose list channels` and `llm-expose list pairs` during preflight.

## Smoke test

1. Start container.
2. Send a test message from an allowed paired chat/channel.
3. Confirm a model response in platform chat.
4. If MCP is configured, trigger one tool-capable prompt and confirm behavior.

## Troubleshooting

- Auth failures from provider: verify API key env vars are set in container environment.
- Channel rejects messages as unpaired: verify `pairs.yaml` includes the exact incoming ID under the right channel name.
- Local model is unreachable: verify model `base_url` network reachability from inside container.
- MCP server failures:
  - stdio transport: verify `command` and `args` are available in image.
  - sse transport: verify URL reachability from container network.

## Upgrades and rollback

1. Build/tag a new image version.
2. Deploy with `docker compose up -d`.
3. Validate smoke test.
4. Roll back by switching image tag back and redeploying.
