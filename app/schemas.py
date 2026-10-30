"""Application request and response schemas for the Local AI Assistant."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
)


class GenerationRequest(BaseModel):
    """Validated application request for LLM text generation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    prompt: str = Field(
        min_length=1,
        description="The input text provided to the language model.",
    )

    format: Literal["json"] | dict[str, Any] | None = Field(
    default=None,
    description="Optional JSON mode or JSON Schema for structured output.",
)

    model: str | None = Field(
        default=None,
        description="Optional model override.",
    )

    stream: StrictBool = Field(
        default=False,
        description="Whether the response should be streamed.",
    )

    max_new_tokens: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=512,
        description="Optional maximum number of generated tokens.",
    )

    num_ctx: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        description="Optional context-window override.",
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional generation temperature.",
    )

    top_k: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        description="Optional top-k sampling setting.",
    )

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional nucleus sampling setting.",
    )

    seed: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional generation seed.",
    )

    stop: list[str] | None = Field(
        default=None,
        description="Optional list of stop sequences.",
    )

    keep_alive: str | None = Field(
        default=None,
        description="Optional model retention override.",
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt_type(cls, value: object) -> str:
        """Require a string prompt with non-whitespace content."""

        if not isinstance(value, str):
            raise ValueError("prompt must be a string.")  # noqa: TRY004

        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only.")

        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        """Reject empty model overrides and surrounding whitespace."""

        if value is None:
            return None

        if not value or value != value.strip():
            raise ValueError(
                "model must be non-empty and contain no surrounding whitespace."
            )

        return value

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Reject empty stop sequences."""

        if value is None:
            return None

        if any(not sequence.strip() for sequence in value):
            raise ValueError("stop sequences must not be empty or whitespace-only.")

        return value

    @field_validator("keep_alive")
    @classmethod
    def validate_keep_alive(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject empty retention overrides and surrounding whitespace."""

        if value is None:
            return None

        if not value or value != value.strip():
            raise ValueError(
                "keep_alive must be non-empty and contain no surrounding whitespace."
            )

        return value


class GenerationResult(BaseModel):
    """Normalized result returned by a runtime adapter."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    text: str = Field(
        description="Generated text returned by the runtime.",
    )

    model: str = Field(
        min_length=1,
        description="Model identifier reported by the runtime.",
    )

    done: bool = Field(
        description="Whether the runtime reports that generation is complete.",
    )

    done_reason: str | None = Field(
        default=None,
        description="Optional generation completion reason reported by the runtime.",
    )

    prompt_eval_count: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional prompt token count reported by the runtime.",
    )

    eval_count: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional generated token count reported by the runtime.",
    )

    total_duration_ns: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional total runtime duration reported in nanoseconds.",
    )
    load_duration_ns: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional model load duration reported in nanoseconds.",
    )

    prompt_eval_duration_ns: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional prompt evaluation duration reported in nanoseconds.",
    )

    eval_duration_ns: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description="Optional output evaluation duration reported in nanoseconds.",
    )

    client_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Client-observed HTTP request latency in milliseconds.",
    )


class StreamEvent(BaseModel):
    """Normalized event emitted during streaming generation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    text_delta: str = Field(
        default="",
        description="The text fragment received from the runtime.",
    )

    event_timestamp: float = Field(
        ge=0,
        description=("Client-observed monotonic timestamp in seconds."),
    )

    is_final: StrictBool = Field(
        default=False,
        description="Whether this event marks the end of the stream.",
    )

    final_metadata: GenerationResult | None = Field(
        default=None,
        description=("Normalized final generation metadata, when available."),
    )
