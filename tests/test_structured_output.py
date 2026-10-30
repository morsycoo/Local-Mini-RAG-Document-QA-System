"""Tests for structured LLM output parsing and validation."""

import pytest
from pydantic import ValidationError

from app.structured_output import (
    AssistantStructuredResponse,
    StructuredOutputParsingError,
    StructuredOutputSchemaError,
    StructuredOutputValidationError,
    parse_and_validate_json,
    parse_and_validate_model,
    parse_json,
    validate_json_schema,
)


def test_parse_valid_json_object() -> None:
    """Parse a valid JSON object into a Python dictionary."""
    result = parse_json('{"answer": "Hello", "confidence": 0.95}')

    assert result == {"answer": "Hello", "confidence": 0.95}


def test_parse_valid_json_array() -> None:
    """Parse a valid JSON array into a Python list."""
    result = parse_json('["Python", "Machine Learning"]')

    assert result == ["Python", "Machine Learning"]


def test_parse_valid_json_string() -> None:
    """Parse a valid JSON string."""
    assert parse_json('"Hello"') == "Hello"


def test_parse_valid_json_null() -> None:
    """Parse the JSON null value."""
    assert parse_json("null") is None


def test_parse_invalid_json() -> None:
    """Reject malformed JSON with a dedicated exception."""
    with pytest.raises(StructuredOutputParsingError):
        parse_json("{invalid json}")


def test_parse_empty_string() -> None:
    """Reject an empty string as invalid JSON."""
    with pytest.raises(StructuredOutputParsingError):
        parse_json("")


def test_parse_non_string_input() -> None:
    """Reject inputs that are not strings."""
    with pytest.raises(TypeError, match="must be a string"):
        parse_json(123)  # type: ignore[arg-type]


def test_validate_json_schema_accepts_valid_data() -> None:
    """Accept data that matches the JSON Schema."""
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    data = {"answer": "Hello", "confidence": 0.95}

    validate_json_schema(data, schema)


def test_validate_json_schema_rejects_invalid_data() -> None:
    """Reject data that violates the JSON Schema."""
    schema = {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["confidence"],
        "additionalProperties": False,
    }

    with pytest.raises(StructuredOutputValidationError):
        validate_json_schema({"confidence": 1.5}, schema)


def test_validate_json_schema_rejects_missing_required_field() -> None:
    """Reject data that omits a required field."""
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    }

    with pytest.raises(StructuredOutputValidationError):
        validate_json_schema({}, schema)


def test_validate_json_schema_rejects_invalid_schema() -> None:
    """Reject an invalid JSON Schema definition."""
    invalid_schema = {"type": "not-a-valid-json-schema-type"}

    with pytest.raises(StructuredOutputSchemaError):
        validate_json_schema({"answer": "Hello"}, invalid_schema)


def test_parse_and_validate_json_accepts_valid_output() -> None:
    """Parse and validate a valid structured response."""
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
        },
        "required": ["answer", "topics"],
        "additionalProperties": False,
    }

    text = '{"answer": "LLMs predict tokens", "topics": ["Inference", "Tokens"]}'

    result = parse_and_validate_json(text, schema)

    assert result == {
        "answer": "LLMs predict tokens",
        "topics": ["Inference", "Tokens"],
    }


def test_parse_and_validate_json_rejects_invalid_output() -> None:
    """Reject valid JSON that violates the expected schema."""
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    }

    with pytest.raises(StructuredOutputValidationError):
        parse_and_validate_json('{"answer": 123}', schema)


def test_structured_response_accepts_valid_data() -> None:
    response = AssistantStructuredResponse(
        answer="Structured output is useful.",
        confidence=0.95,
    )

    assert response.answer == "Structured output is useful."
    assert response.confidence == 0.95


def test_structured_response_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        AssistantStructuredResponse(
            answer="Example",
            confidence=1.5,
        )


def test_structured_response_rejects_empty_answer() -> None:
    with pytest.raises(ValidationError):
        AssistantStructuredResponse(
            answer="",
            confidence=0.5,
        )


def test_structured_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AssistantStructuredResponse(
            answer="Example",
            confidence=0.5,
            unexpected_field="not allowed",
        )


def test_parse_and_validate_model_accepts_valid_data() -> None:
    result = parse_and_validate_model(
        '{"answer": "Hello", "confidence": 0.95}',
        AssistantStructuredResponse,
    )

    assert isinstance(result, AssistantStructuredResponse)
    assert result.answer == "Hello"
    assert result.confidence == 0.95


def test_parse_and_validate_model_rejects_invalid_data() -> None:
    with pytest.raises(StructuredOutputValidationError):
        parse_and_validate_model(
            '{"answer": "Hello", "confidence": 1.5}',
            AssistantStructuredResponse,
        )
