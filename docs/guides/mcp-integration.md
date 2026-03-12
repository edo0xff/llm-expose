# MCP Integration

You can attach one or more MCP servers to a channel so model completions can access external tools through the MCP runtime.

## Typical flow

1. Add MCP server config.
2. Attach server to a channel.
3. Run the channel and trigger tool-aware requests.

Use the CLI `add mcp`, `list mcp`, and `delete mcp` commands to manage MCP definitions.

## Add MCP servers

### Stdio transport

```bash
llm-expose add mcp \
	--name web-search \
	--transport stdio \
	--command uvx \
	--args mcp-server-web-search \
	--tool-confirmation never \
	-y --no-input
```

### SSE transport

```bash
llm-expose add mcp \
	--name remote-tools \
	--transport sse \
	--url http://mcp.internal:3000/sse \
	--tool-confirmation required \
	-y --no-input
```

CLI note:

- `add mcp` currently supports `stdio` and `sse` transports.
- `builtin-core` tools are provided by runtime internals and can be attached by name on channels.

## Attach MCP servers to channels

```bash
llm-expose add channel \
	--name ops-discord \
	--client-type discord \
	--bot-token "YOUR_DISCORD_BOT_TOKEN" \
	--model-name gpt4o-mini \
	--mcp-server builtin-core \
	--mcp-server web-search \
	--mcp-server remote-tools \
	-y --no-input
```

If any referenced MCP server does not exist, headless channel creation fails with a clear error.

## Confirmation and safety model

MCP confirmation works at two levels:

- Global setting in `mcp_servers.yaml` (`settings.confirmation_mode`: `optional` or `required`).
- Per-server override (`tool_confirmation`: `default`, `required`, `never`).

Practical policy patterns:

- Low friction: global `optional`, read-only tools with per-server `never`.
- Safer operations: global `required`, and keep risky servers as `required`.

## Example MCP YAML

```yaml
settings:
	confirmation_mode: required
	tool_timeout_seconds: 60
	expose_attachment_paths: false
servers:
	- name: web-search
		transport: stdio
		command: uvx
		args: [mcp-server-web-search]
		enabled: true
		tool_confirmation: never
	- name: remote-tools
		transport: sse
		url: http://mcp.internal:3000/sse
		enabled: true
		tool_confirmation: required
		allowed_tools: [ticket_create, ticket_get]
```

## Runtime behavior notes

- MCP runtime initializes lazily on first relevant request.
- Tools from disabled servers are not exposed.
- Timeout is enforced per tool call via `tool_timeout_seconds`.

## Verify MCP setup

```bash
llm-expose list mcp
llm-expose list channels
```

Then start the channel and send a tool-relevant prompt.

## Troubleshooting

- No tools appear: verify server names are attached on the channel config.
- Stdio server fails: verify `command` and `args` exist in runtime environment.
- SSE server fails: verify URL/network reachability.
- Tool runs are blocked unexpectedly: verify global vs per-server confirmation settings.
