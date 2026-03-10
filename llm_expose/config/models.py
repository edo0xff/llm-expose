"""Configuration models for llm-expose using Pydantic."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider_name: str = Field(
        description="Provider name, e.g. 'openai', 'local', 'anthropic'"
    )
    model: str = Field(description="Model identifier, e.g. 'gpt-4o', 'llama3'")
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider (not required for local models)",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for local or self-hosted models (e.g. LM Studio, Ollama proxy)",
    )
    supports_vision: Optional[bool] = Field(
        default=None,
        description=(
            "Override model vision capability detection. "
            "When unset, provider attempts auto-detection."
        ),
    )

    @field_validator("provider_name", "model")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        """Ensure required string fields are not blank."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace")
        return v.strip()


class TelegramClientConfig(BaseModel):
    """Configuration for the Telegram client adapter."""

    client_type: Literal["telegram"] = "telegram"
    bot_token: str = Field(description="Telegram bot token from @BotFather")
    mcp_servers: list[str] = Field(
        default_factory=list,
        description="List of MCP server names attached to this channel",
    )
    system_prompt_path: Optional[str] = Field(
        default=None,
        description="Path to a file containing the custom system prompt for this channel. Falls back to default when omitted.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Default LLM model name for this channel (used with --llm-completion)",
    )

    @field_validator("bot_token")
    @classmethod
    def token_must_not_be_empty(cls, v: str) -> str:
        """Ensure bot token is not blank."""
        if not v or not v.strip():
            raise ValueError("bot_token must not be empty or whitespace")
        return v.strip()

    @field_validator("mcp_servers")
    @classmethod
    def normalize_mcp_servers(cls, values: list[str]) -> list[str]:
        """Normalize attached MCP server names preserving order and uniqueness."""
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            name = value.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    @field_validator("system_prompt_path")
    @classmethod
    def validate_system_prompt_path(cls, value: Optional[str]) -> Optional[str]:
        """Validate system prompt path by trimming outer whitespace."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: Optional[str]) -> Optional[str]:
        """Normalize optional model name by trimming outer whitespace."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server integration."""

    name: str = Field(description="Unique MCP server name")
    transport: Literal["stdio", "sse", "http", "builtin"] = Field(
        default="stdio", description="Transport type used to connect to the MCP server"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to run for stdio transport (e.g. 'npx', 'uvx')",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments passed to the command for stdio transport",
    )
    url: Optional[str] = Field(
        default=None,
        description="Server URL for SSE transport",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables passed to the MCP server process",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Optional allow-list of MCP tool names available to the model",
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")
    tool_confirmation: Literal["required", "never", "default"] = Field(
        default="default",
        description="Tool confirmation mode: 'required' forces approval, 'never' auto-executes, 'default' uses global setting",
    )

    @field_validator("name")
    @classmethod
    def server_name_must_not_be_empty(cls, v: str) -> str:
        """Ensure server name is not blank."""
        if not v or not v.strip():
            raise ValueError("MCP server name must not be empty or whitespace")
        return v.strip()

    @field_validator("command")
    @classmethod
    def command_must_not_be_empty_when_present(cls, v: Optional[str]) -> Optional[str]:
        """Normalize optional command by trimming whitespace."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty_when_present(cls, v: Optional[str]) -> Optional[str]:
        """Normalize optional URL by trimming whitespace."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None


class MCPSettingsConfig(BaseModel):
    """Global MCP runtime settings shared by all exposures."""

    confirmation_mode: Literal["required", "optional"] = Field(
        default="optional",
        description="Whether tool calls require explicit user confirmation",
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Max tool execution time in seconds",
    )


class MCPConfig(BaseModel):
    """Top-level MCP configuration persisted on disk."""

    settings: MCPSettingsConfig = Field(default_factory=MCPSettingsConfig)
    servers: list[MCPServerConfig] = Field(default_factory=list)


class PairingsConfig(BaseModel):
    """Top-level pairing configuration persisted on disk.

    Pair IDs are stored by channel config name so each configured channel can
    have an independent allowlist of sender/channel identifiers.
    """

    pairs_by_channel: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("pairs_by_channel")
    @classmethod
    def normalize_pairs_by_channel(
        cls, values: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Trim and deduplicate channel names and pair IDs."""
        normalized: dict[str, list[str]] = {}
        for raw_channel_name, raw_pair_ids in values.items():
            channel_name = raw_channel_name.strip()
            if not channel_name:
                continue

            cleaned_ids: list[str] = []
            seen_ids: set[str] = set()
            for raw_pair_id in raw_pair_ids:
                pair_id = raw_pair_id.strip()
                if not pair_id or pair_id in seen_ids:
                    continue
                seen_ids.add(pair_id)
                cleaned_ids.append(pair_id)

            normalized[channel_name] = cleaned_ids
        return normalized


# TODO: Add DiscordClientConfig when Discord client is implemented
# TODO: Add SlackClientConfig when Slack client is implemented

ClientConfig = Annotated[
    Union[TelegramClientConfig],
    Field(discriminator="client_type"),
]


class ExposureConfig(BaseModel):
    """Top-level configuration for a single LLM exposure."""

    name: str = Field(description="Unique name for this exposure configuration")
    channel_name: Optional[str] = Field(
        default=None,
        description="Selected saved channel config name used for pair scoping.",
    )
    provider: ProviderConfig = Field(description="LLM provider settings")
    client: TelegramClientConfig = Field(description="Messaging client settings")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Ensure name is a non-empty, cross-platform filesystem-safe identifier.

        The forbidden set covers characters that are invalid in file names on
        Windows (``/ \\ : * ? " < > |``) so configs remain portable across
        operating systems.
        """
        if not v or not v.strip():
            raise ValueError("Exposure name must not be empty or whitespace")
        stripped = v.strip()
        forbidden = set('/\\:*?"<>|')
        if any(c in forbidden for c in stripped):
            raise ValueError(
                f"Exposure name contains forbidden characters: {forbidden}"
            )
        return stripped

    @field_validator("channel_name")
    @classmethod
    def normalize_channel_name(cls, value: Optional[str]) -> Optional[str]:
        """Normalize optional channel namespace by trimming whitespace."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
