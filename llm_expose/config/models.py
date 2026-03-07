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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")

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

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Ensure name is a non-empty, filesystem-safe identifier."""
        if not v or not v.strip():
            raise ValueError("Exposure name must not be empty or whitespace")
        stripped = v.strip()
        forbidden = set('/\\:*?"<>|')
        if any(c in forbidden for c in stripped):
            raise ValueError(
                f"Exposure name contains forbidden characters: {forbidden}"
            )
        return stripped
