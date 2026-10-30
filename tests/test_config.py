"""Tests for centralized runtime configuration."""

import pytest
from pydantic import ValidationError

from app.config import RuntimeSettings, load_settings


def test_default_settings() -> None:
    """Verify the project's default configuration."""

    settings = load_settings(environ={})

    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.model_name == "qwen3.8:27b"
    assert settings.request_timeout_seconds == 120.0
    assert settings.keep_alive == "5m"
    assert settings.num_ctx == 4096
    assert settings.max_new_tokens == 512
    assert settings.temperature == 0.0

    assert settings.top_k is None
    assert settings.top_p is None
    assert settings.seed is None


def test_environment_overrides() -> None:
    """Verify that explicit environment values override defaults."""

    settings = load_settings(
        environ={
            "OLLAMA_HOST": "http://localhost:11434",
            "MODEL_NAME": "qwen3.8:27b",
            "REQUEST_TIMEOUT_SECONDS": "45",
            "NUM_CTX": "4096",
            "MAX_NEW_TOKENS": "24",
            "TEMPERATURE": "0.2",
            "TOP_K": "20",
            "TOP_P": "0.9",
            "SEED": "42",
        }
    )

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.request_timeout_seconds == 45.0
    assert settings.num_ctx == 4096
    assert settings.max_new_tokens == 24
    assert settings.temperature == 0.2
    assert settings.top_k == 20
    assert settings.top_p == 0.9
    assert settings.seed == 42


def test_env_file_and_environment_precedence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that process environment values override values from .env."""

    environment_keys = [
        "OLLAMA_HOST",
        "MODEL_NAME",
        "MAX_REQUEST_BODY_BYTES",
        "MAX_MESSAGE_CHARS",
        "MAX_ACTIVE_SESSIONS",
        "MAX_CONVERSATIONS_PER_SESSION",
        "SESSION_IDLE_TTL_SECONDS",
        "SESSION_ABSOLUTE_TTL_SECONDS",
        "SESSION_RATE_LIMIT_REQUESTS",
        "SESSION_RATE_LIMIT_WINDOW_SECONDS",
        "GLOBAL_GENERATION_RATE_LIMIT_REQUESTS",
        "GLOBAL_GENERATION_RATE_LIMIT_WINDOW_SECONDS",
        "MAX_ACTIVE_GENERATIONS",
        "REQUEST_TIMEOUT_SECONDS",
        "KEEP_ALIVE",
        "NUM_CTX",
        "MAX_NEW_TOKENS",
        "TEMPERATURE",
        "TOP_K",
        "TOP_P",
        "SEED",
    ]

    # Remove potentially conflicting settings from the current process.
    for key in environment_keys:
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"

    env_file.write_text(
        "\n".join(
            [
                "OLLAMA_HOST=http://localhost:11434",
                "MODEL_NAME=qwen3.8:27b",
                "MAX_NEW_TOKENS=20",
            ]
        ),
        encoding="utf-8",
    )

    # The operating-system environment must take precedence over .env.
    monkeypatch.setenv("MAX_NEW_TOKENS", "32")

    settings = load_settings(env_file=env_file)

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.model_name == "qwen3.8:27b"
    assert settings.max_new_tokens == 32


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", -1),
        ("num_ctx", 0),
        ("max_new_tokens", 0),
        ("max_new_tokens", 513),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_k", 0),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("seed", -1),
    ],
)
def test_invalid_numeric_values_are_rejected(
    field_name: str,
    invalid_value: int | float,
) -> None:
    """Reject numeric settings that violate the configuration policy."""

    with pytest.raises(ValidationError):
        RuntimeSettings(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "invalid_host",
    [
        "localhost:11434",
        "ftp://localhost:11434",
        "http://",
        "http://localhost:70000",
        "http://user:password@localhost:11434",
        "http://localhost:11434/api",
    ],
)
def test_invalid_ollama_hosts_are_rejected(
    invalid_host: str,
) -> None:
    """Reject URLs that violate the configured Ollama base URL policy."""

    with pytest.raises(ValidationError):
        RuntimeSettings(ollama_host=invalid_host)


@pytest.mark.parametrize(
    "invalid_model_name",
    [
        "",
        " ",
        " qwen3.8:27b",
        "qwen3.8:27b ",
    ],
)
def test_invalid_model_names_are_rejected(
    invalid_model_name: str,
) -> None:
    """Reject empty model identifiers and surrounding whitespace."""

    with pytest.raises(ValidationError):
        RuntimeSettings(model_name=invalid_model_name)


def test_unknown_configuration_fields_are_rejected() -> None:
    """Reject unsupported fields rather than silently accepting them."""

    with pytest.raises(ValidationError):
        RuntimeSettings(unsupported_option="unexpected")

def test_allowed_origins_configuration() -> None:
    """Verify allowed origins default and environment override."""

    defaults = load_settings(environ={})
    assert defaults.allowed_origins == ""

    settings = load_settings(
        environ={
            "ALLOWED_ORIGINS": "http://127.0.0.1:8000,http://localhost:8000"
        }
    )
    assert settings.allowed_origins == (
        "http://127.0.0.1:8000,http://localhost:8000"
    )


def test_security_settings_load_from_env_file_and_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify security settings load from .env and environment overrides."""

    monkeypatch.delenv("COOKIE_SECURE", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "COOKIE_SECURE=true",
                "ALLOWED_ORIGINS=http://localhost:8000",
            ]
        ),
        encoding="utf-8",
    )

    file_settings = load_settings(env_file=env_file)

    assert file_settings.cookie_secure is True
    assert file_settings.allowed_origins == "http://localhost:8000"

    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://example.com",
    )

    overridden_settings = load_settings(env_file=env_file)

    assert overridden_settings.cookie_secure is False
    assert overridden_settings.allowed_origins == "https://example.com"


def test_public_mode_defaults_to_false() -> None:
    """Keep public mode disabled by default."""

    settings = load_settings(environ={})

    assert settings.public_mode is False


def test_public_mode_loads_from_env_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load public mode from .env when no explicit environment is injected."""

    monkeypatch.delenv("PUBLIC_MODE", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("PUBLIC_MODE=true" + chr(10), encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert settings.public_mode is True


def test_public_mode_environment_overrides_env_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow the process environment to override the .env value."""

    env_file = tmp_path / ".env"
    env_file.write_text("PUBLIC_MODE=true" + chr(10), encoding="utf-8")

    monkeypatch.setenv("PUBLIC_MODE", "false")

    settings = load_settings(env_file=env_file)

    assert settings.public_mode is False


def test_validate_public_mode_settings_keeps_local_mode_unchanged() -> None:
    """Local mode must not require public deployment settings."""

    from app.config import validate_public_mode_settings

    validate_public_mode_settings(RuntimeSettings())


def test_validate_public_mode_settings_accepts_secure_https_origin() -> None:
    """Accept public mode with secure cookies and an HTTPS origin."""

    from app.config import validate_public_mode_settings

    settings = RuntimeSettings(
        public_mode=True,
        cookie_secure=True,
        allowed_origins="https://example.com",
    )

    validate_public_mode_settings(settings)


@pytest.mark.parametrize(
    ("cookie_secure", "allowed_origins"),
    [
        (False, "https://example.com"),
        (True, ""),
        (True, "http://example.com"),
        (True, "not-a-valid-origin"),
        (True, "https://example.com/path"),
        (True, "https://user@example.com"),
    ],
)
def test_validate_public_mode_settings_rejects_unsafe_configuration(
    cookie_secure: bool,
    allowed_origins: str,
) -> None:
    """Reject incomplete or unsafe public-mode settings."""

    from app.config import validate_public_mode_settings

    settings = RuntimeSettings(
        public_mode=True,
        cookie_secure=cookie_secure,
        allowed_origins=allowed_origins,
    )

    with pytest.raises(ValueError):
        validate_public_mode_settings(settings)


def test_web_resource_limits_have_safe_defaults() -> None:
    settings = load_settings(environ={})
    assert settings.max_request_body_bytes == 65536
    assert settings.max_message_chars == 4000
    assert settings.max_active_sessions == 25
    assert settings.max_conversations_per_session == 5
    assert settings.session_idle_ttl_seconds == 900
    assert settings.session_absolute_ttl_seconds == 14400
    assert settings.session_rate_limit_requests == 10
    assert settings.session_rate_limit_window_seconds == 60
    assert settings.global_generation_rate_limit_requests == 20
    assert settings.global_generation_rate_limit_window_seconds == 60
    assert settings.max_active_generations == 1


def test_public_mode_rejects_unsafe_resource_limits() -> None:
    from app.config import validate_public_mode_settings

    settings = RuntimeSettings(
        public_mode=True,
        cookie_secure=True,
        allowed_origins="https://example.com",
        max_active_sessions=26,
    )
    with pytest.raises(ValueError, match="bounded request, message, session"):
        validate_public_mode_settings(settings)


def test_resource_limit_environment_overrides_are_loaded() -> None:
    settings = load_settings(environ={
        "MAX_REQUEST_BODY_BYTES": "32768",
        "MAX_MESSAGE_CHARS": "2000",
        "MAX_ACTIVE_SESSIONS": "10",
        "MAX_CONVERSATIONS_PER_SESSION": "3",
        "SESSION_IDLE_TTL_SECONDS": "600",
        "SESSION_ABSOLUTE_TTL_SECONDS": "3600",
        "SESSION_RATE_LIMIT_REQUESTS": "5",
        "SESSION_RATE_LIMIT_WINDOW_SECONDS": "60",
        "GLOBAL_GENERATION_RATE_LIMIT_REQUESTS": "12",
        "GLOBAL_GENERATION_RATE_LIMIT_WINDOW_SECONDS": "60",
        "MAX_ACTIVE_GENERATIONS": "1",
    })
    assert settings.max_request_body_bytes == 32768
    assert settings.max_message_chars == 2000
    assert settings.max_active_sessions == 10
    assert settings.max_conversations_per_session == 3
    assert settings.session_idle_ttl_seconds == 600
    assert settings.session_absolute_ttl_seconds == 3600
    assert settings.session_rate_limit_requests == 5
    assert settings.session_rate_limit_window_seconds == 60
    assert settings.global_generation_rate_limit_requests == 12
    assert settings.global_generation_rate_limit_window_seconds == 60
    assert settings.max_active_generations == 1
