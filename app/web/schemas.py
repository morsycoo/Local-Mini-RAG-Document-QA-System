"""Pydantic schemas for the HADOTA web API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictRequestModel(BaseModel):
    """Base model that rejects unexpected request fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )


class CreateConversationRequest(StrictRequestModel):
    """Request to create a new conversation."""

    title: str | None = Field(
        default=None,
        max_length=100,
        description="Optional display title for the conversation.",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must not be empty or whitespace-only.")
        return value


class SendMessageRequest(StrictRequestModel):
    """Request to send a user message."""

    content: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's message.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("content must be a string.")
        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only.")
        return value


class SessionResponse(BaseModel):
    """Public session creation response."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ConversationResponse(BaseModel):
    """Public conversation creation response."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    title: str | None = None


class MessageResponse(BaseModel):
    """Public response containing a completed assistant message."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    role: str = "assistant"
    content: str
    title: str | None = None
    done: bool = True
    continuation_available: bool = False

