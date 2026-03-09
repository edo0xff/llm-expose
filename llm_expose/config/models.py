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

    @field_validator("bot_token")
    @classmethod
    def token_must_not_be_empty(cls, v: str) -> str:
        """Ensure bot token is not blank."""
        if not v or not v.strip():
            raise ValueError("bot_token must not be empty or whitespace")
        return v.strip()


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server integration."""

    name: str = Field(description="Unique MCP server name")
    transport: Literal["stdio", "sse", "http"] = Field(
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


# TODO: Add DiscordClientConfig when Discord client is implemented
# TODO: Add SlackClientConfig when Slack client is implemented

ClientConfig = Annotated[
    Union[TelegramClientConfig],
    Field(discriminator="client_type"),
]


class ExposureConfig(BaseModel):
    """Top-level configuration for a single LLM exposure."""

    name: str = Field(description="Unique name for this exposure configuration")
    provider: ProviderConfig = Field(description="LLM provider settings")
    client: TelegramClientConfig = Field(description="Messaging client settings")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for this exposure. If not set, uses default prompt.",
    )

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
