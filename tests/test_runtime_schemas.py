"""Tests for application generation request schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import GenerationRequest


def test_minimal_generation_request() -> None:
    """Accept a valid prompt while leaving optional overrides unset."""

    request = GenerationRequest(prompt="Explain how an LLM runtime works.")

    assert request.prompt == "Explain how an LLM runtime works."
    assert request.format is None
    assert request.model is None
    assert request.stream is False
    assert request.max_new_tokens is None
    assert request.num_ctx is None
    assert request.temperature is None
    assert request.top_k is None
    assert request.top_p is None
    assert request.seed is None
    assert request.stop is None
    assert request.keep_alive is None


def test_generation_request_with_explicit_options() -> None:
    """Accept valid per-request generation overrides."""

    request = GenerationRequest(
        prompt="Explain runtime adapters.",
        format="json",
        model="qwen3.8:27b",
        stream=True,
        max_new_tokens=32,
        num_ctx=4096,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        seed=42,
        stop=["END_OF_RESPONSE"],
        keep_alive="5m",
    )

    assert request.format == "json"
    assert request.model == "qwen3.8:27b"
    assert request.stream is True
    assert request.max_new_tokens == 32
    assert request.num_ctx == 4096
    assert request.temperature == 0.2
    assert request.top_k == 20
    assert request.top_p == 0.9
    assert request.seed == 42
    assert request.stop == ["END_OF_RESPONSE"]
    assert request.keep_alive == "5m"


@pytest.mark.parametrize(
    "invalid_prompt",
    [
        "",
        " ",
        "\n\t",
        None,
        123,
    ],
)
def test_invalid_prompts_are_rejected(
    invalid_prompt: object,
) -> None:
    """Reject prompts that violate the text input contract."""

    with pytest.raises(ValidationError):
        GenerationRequest(prompt=invalid_prompt)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_new_tokens", 0),
        ("max_new_tokens", 513),
        ("max_new_tokens", 1.5),
        ("num_ctx", 0),
        ("num_ctx", -1),
        ("num_ctx", 4096.0),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_k", 0),
        ("top_k", -1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("seed", -1),
        ("seed", 42.5),
    ],
)
def test_invalid_generation_options_are_rejected(
    field_name: str,
    invalid_value: int | float,
) -> None:
    """Reject invalid numeric values and types."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            **{field_name: invalid_value},
        )


@pytest.mark.parametrize(
    "invalid_stream",
    [
        "true",
        "false",
        1,
        0,
        None,
    ],
)
def test_stream_requires_a_boolean(
    invalid_stream: object,
) -> None:
    """Reject non-Boolean values for the stream field."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            stream=invalid_stream,
        )


@pytest.mark.parametrize(
    "invalid_stop",
    [
        [""],
        ["   "],
        ["VALID", ""],
    ],
)
def test_invalid_stop_sequences_are_rejected(
    invalid_stop: list[str],
) -> None:
    """Reject empty stop sequences."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            stop=invalid_stop,
        )


@pytest.mark.parametrize(
    "invalid_model",
    [
        "",
        " ",
        " qwen3.8:27b",
        "qwen3.8:27b ",
    ],
)
def test_invalid_model_overrides_are_rejected(
    invalid_model: str,
) -> None:
    """Reject invalid model overrides."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            model=invalid_model,
        )


def test_unknown_request_fields_are_rejected() -> None:
    """Reject fields outside the supported request contract."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            unsupported_option="unexpected",
        )


def test_request_serialization() -> None:
    """Serialize a valid request into a predictable dictionary."""

    request = GenerationRequest(
        prompt="Explain configuration validation.",
        max_new_tokens=24,
    )

    payload = request.model_dump()

    assert payload["prompt"] == "Explain configuration validation."
    assert payload["format"] is None
    assert payload["max_new_tokens"] == 24
    assert payload["stream"] is False
    assert payload["temperature"] is None


@pytest.mark.parametrize(
    "invalid_format",
    [
        "xml",
        "text",
        "",
        123,
    ],
)
def test_invalid_output_formats_are_rejected(
    invalid_format: object,
) -> None:
    """Reject unsupported structured output format values."""

    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="Test input.",
            format=invalid_format,
        )


def test_generation_request_accepts_json_schema() -> None:
    """Accept a JSON Schema as the structured output format."""

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    request = GenerationRequest(
        prompt="Answer the question.",
        format=schema,
    )

    assert request.format == schema
