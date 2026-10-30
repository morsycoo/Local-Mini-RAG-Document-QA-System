"""Tests for normalized generation results."""

import pytest
from pydantic import ValidationError

from app.schemas import GenerationResult


def test_minimal_generation_result() -> None:
    """Accept a result when only required fields are available."""

    result = GenerationResult(
        text="Local inference runs on the configured runtime.",
        model="qwen3.8:27b",
        done=True,
    )

    assert result.text == "Local inference runs on the configured runtime."
    assert result.model == "qwen3.8:27b"
    assert result.done is True

    assert result.done_reason is None
    assert result.prompt_eval_count is None
    assert result.eval_count is None
    assert result.total_duration_ns is None


def test_generation_result_with_runtime_metadata() -> None:
    """Preserve runtime metadata when the service reports it."""

    result = GenerationResult(
        text="A normalized response.",
        model="qwen3.8:27b",
        done=True,
        done_reason="stop",
        prompt_eval_count=12,
        eval_count=24,
        total_duration_ns=1_500_000_000,
    )

    assert result.done_reason == "stop"
    assert result.prompt_eval_count == 12
    assert result.eval_count == 24
    assert result.total_duration_ns == 1_500_000_000


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("prompt_eval_count", -1),
        ("eval_count", -1),
        ("total_duration_ns", -1),
    ],
)
def test_negative_metadata_values_are_rejected(
    field_name: str,
    invalid_value: int,
) -> None:
    """Reject negative counters and durations."""

    with pytest.raises(ValidationError):
        GenerationResult(
            text="Test response.",
            model="qwen3.8:27b",
            done=True,
            **{field_name: invalid_value},
        )


def test_empty_model_identifier_is_rejected() -> None:
    """Require a non-empty model identifier."""

    with pytest.raises(ValidationError):
        GenerationResult(
            text="Test response.",
            model="",
            done=True,
        )


def test_unknown_result_fields_are_rejected() -> None:
    """Reject fields outside the normalized result contract."""

    with pytest.raises(ValidationError):
        GenerationResult(
            text="Test response.",
            model="qwen3.8:27b",
            done=True,
            unsupported_field="unexpected",
        )


def test_result_serialization() -> None:
    """Serialize normalized output into a predictable dictionary."""

    result = GenerationResult(
        text="Normalized output.",
        model="qwen3.8:27b",
        done=True,
        eval_count=8,
    )

    payload = result.model_dump()

    assert payload["text"] == "Normalized output."
    assert payload["model"] == "qwen3.8:27b"
    assert payload["done"] is True
    assert payload["eval_count"] == 8
    assert payload["total_duration_ns"] is None
