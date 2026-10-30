"""Centralized runtime configuration for the Local AI Assistant."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuntimeSettings(BaseModel):
    """Validated runtime and generation settings."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    ollama_host: str = "http://127.0.0.1:11434"
    model_name: str = "qwen3.8:27b"
    cookie_secure: bool = False
    allowed_origins: str = ""
    public_mode: bool = False

    # Web API resource-safety limits. Public mode is deliberately bounded.
    max_request_body_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_message_chars: int = Field(default=4000, ge=1, le=4000)
    max_active_sessions: int = Field(default=25, ge=1, le=100)
    max_conversations_per_session: int = Field(default=5, ge=1, le=20)
    session_idle_ttl_seconds: int = Field(default=900, ge=60, le=7200)
    session_absolute_ttl_seconds: int = Field(default=14400, ge=300, le=86400)
    session_rate_limit_requests: int = Field(default=10, ge=1, le=120)
    session_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    global_generation_rate_limit_requests: int = Field(default=20, ge=1, le=120)
    global_generation_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    max_active_generations: int = Field(default=1, ge=1, le=1)

    request_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
    )

    keep_alive: str = "5m"
    think: bool | Literal["low", "medium", "xhigh"] = False

    num_ctx: int = Field(
        default=4096,
        ge=1,
    )

    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=512,
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
    )

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    seed: int | None = Field(
        default=None,
        ge=0,
    )

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        """Validate the Ollama base URL."""

        if value != value.strip():
            raise ValueError("ollama_host must not contain surrounding whitespace.")

        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError("ollama_host must be a valid URL.") from error

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("ollama_host must use HTTP or HTTPS.")

        if not parsed.hostname:
            raise ValueError("ollama_host must contain a hostname.")

        if port is not None and not 1 <= port <= 65535:
            raise ValueError("ollama_host contains an invalid port.")

        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in ollama_host.")

        if parsed.path not in {"", "/"}:
            raise ValueError("ollama_host must not contain an API endpoint path.")

        if parsed.query or parsed.fragment:
            raise ValueError("ollama_host must not contain a query or fragment.")

        return value.rstrip("/")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Reject empty model identifiers and surrounding whitespace."""

        if not value or value != value.strip():
            raise ValueError(
                "model_name must be non-empty and contain no surrounding whitespace."
            )

        return value

    @field_validator("keep_alive")
    @classmethod
    def validate_keep_alive(cls, value: str) -> str:
        """Require a non-empty model retention setting."""

        if not value or value != value.strip():
            raise ValueError(
                "keep_alive must be non-empty and contain no surrounding whitespace."
            )

        return value


ENV_FIELD_MAPPING = {
    "OLLAMA_HOST": "ollama_host",
    "MODEL_NAME": "model_name",
    "COOKIE_SECURE": "cookie_secure",
    "ALLOWED_ORIGINS": "allowed_origins",
    "PUBLIC_MODE": "public_mode",
    "MAX_REQUEST_BODY_BYTES": "max_request_body_bytes",
    "MAX_MESSAGE_CHARS": "max_message_chars",
    "MAX_ACTIVE_SESSIONS": "max_active_sessions",
    "MAX_CONVERSATIONS_PER_SESSION": "max_conversations_per_session",
    "SESSION_IDLE_TTL_SECONDS": "session_idle_ttl_seconds",
    "SESSION_ABSOLUTE_TTL_SECONDS": "session_absolute_ttl_seconds",
    "SESSION_RATE_LIMIT_REQUESTS": "session_rate_limit_requests",
    "SESSION_RATE_LIMIT_WINDOW_SECONDS": "session_rate_limit_window_seconds",
    "GLOBAL_GENERATION_RATE_LIMIT_REQUESTS": "global_generation_rate_limit_requests",
    "GLOBAL_GENERATION_RATE_LIMIT_WINDOW_SECONDS": "global_generation_rate_limit_window_seconds",
    "MAX_ACTIVE_GENERATIONS": "max_active_generations",
    "REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "KEEP_ALIVE": "keep_alive",
    "NUM_CTX": "num_ctx",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "TEMPERATURE": "temperature",
    "TOP_K": "top_k",
    "TOP_P": "top_p",
    "SEED": "seed",
    "THINK": "think",
}


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> RuntimeSettings:
    """Load validated settings from defaults, a .env file, and environment variables.

    Precedence:
        1. Model defaults.
        2. Values loaded from .env.
        3. Operating-system environment variables.

    Passing ``environ`` explicitly creates an isolated configuration source,
    which is useful for deterministic tests.
    """

    raw_values: dict[str, str | int | float] = {}

    if environ is None:
        selected_env_file = (
            Path(env_file)
            if env_file is not None
            else Path(__file__).resolve().parents[1] / ".env"
        )

        if selected_env_file.is_file():
            file_values = dotenv_values(selected_env_file)

            raw_values.update(
                {
                    key: value
                    for key, value in file_values.items()
                    if value is not None and key in ENV_FIELD_MAPPING
                }
            )

        # Operating-system variables override values from .env.
        raw_values.update(
            {
                key: value
                for key, value in os.environ.items()
                if key in ENV_FIELD_MAPPING
            }
        )

    else:
        # Explicit injection prevents machine settings from affecting tests.
        raw_values.update(
            {key: value for key, value in environ.items() if key in ENV_FIELD_MAPPING}
        )

    settings_values = {
        field_name: raw_values[environment_name]
        for environment_name, field_name in ENV_FIELD_MAPPING.items()
        if environment_name in raw_values
    }

    return RuntimeSettings.model_validate(settings_values)


def validate_public_mode_settings(settings: RuntimeSettings) -> None:
    """Require essential cookie and origin settings for public mode."""

    if not settings.public_mode:
        return

    if not settings.cookie_secure:
        raise ValueError(
            "PUBLIC_MODE requires COOKIE_SECURE=true."
        )

    # These constraints are intentionally stricter than local development.
    # They prevent a public configuration from disabling or greatly weakening
    # the core resource protections for this local-machine inference service.
    public_limits = (
        settings.max_request_body_bytes <= 65536,
        settings.max_message_chars <= 4000,
        settings.max_active_sessions <= 25,
        settings.max_conversations_per_session <= 5,
        settings.session_idle_ttl_seconds <= 900,
        settings.session_absolute_ttl_seconds <= 14400,
        settings.session_rate_limit_requests <= 10,
        settings.session_rate_limit_window_seconds >= 60,
        settings.global_generation_rate_limit_requests <= 20,
        settings.global_generation_rate_limit_window_seconds >= 60,
        settings.max_active_generations == 1,
    )
    if not all(public_limits):
        raise ValueError(
            "PUBLIC_MODE requires bounded request, message, session, rate, "
            "lifetime, and single-generation limits."
        )

    origins = [
        item.strip()
        for item in settings.allowed_origins.split(",")
        if item.strip()
    ]

    if not origins:
        raise ValueError(
            "PUBLIC_MODE requires at least one ALLOWED_ORIGINS entry."
        )

    for origin in origins:
        try:
            parsed = urlsplit(origin)
            port = parsed.port
        except ValueError as error:
            raise ValueError(
                "PUBLIC_MODE requires valid HTTP(S) origins."
            ) from error

        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(
                "PUBLIC_MODE requires valid HTTP(S) origins without paths, "
                "credentials, queries, or fragments."
            )

        if parsed.scheme.lower() != "https":
            raise ValueError(
                "PUBLIC_MODE requires HTTPS origins."
            )
