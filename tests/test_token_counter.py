"""Tests for the local token counter."""

from pathlib import Path

import pytest

from app.context import TokenCounter

TOKENIZER_PATH = Path("assets/tokenizers/qwen3.8-27b/tokenizer.json")


@pytest.fixture
def token_counter() -> TokenCounter:
    """Create a token counter using the optional large local tokenizer asset."""
    if not TOKENIZER_PATH.is_file():
        pytest.skip(
            "Local tokenizer JSON is not included in this checkout; "
            "provide assets/tokenizers/qwen3.8-27b/tokenizer.json to run these tests."
        )
    return TokenCounter(TOKENIZER_PATH)


def test_token_counter_loads_local_tokenizer(
    token_counter: TokenCounter,
) -> None:
    """The token counter should load the tokenizer successfully."""
    assert token_counter.count("Hello world") > 0


def test_token_counter_counts_empty_text(
    token_counter: TokenCounter,
) -> None:
    """Empty text should produce zero tokens."""
    assert token_counter.count("") == 0


def test_token_counter_counts_english_text(
    token_counter: TokenCounter,
) -> None:
    """English text should produce a positive token count."""
    assert token_counter.count("Explain how a Transformer processes tokens.") == 8


def test_token_counter_rejects_non_string_input(
    token_counter: TokenCounter,
) -> None:
    """Non-string input should raise a TypeError."""
    with pytest.raises(TypeError, match="text must be a string"):
        token_counter.count(123)  # type: ignore[arg-type]


def test_token_counter_rejects_missing_tokenizer_file(
    tmp_path: Path,
) -> None:
    """A missing tokenizer file should raise FileNotFoundError."""
    missing_path = tmp_path / "missing-tokenizer.json"

    with pytest.raises(FileNotFoundError, match="Tokenizer file was not found"):
        TokenCounter(missing_path)


def test_token_counter_is_callable(
    token_counter: TokenCounter,
) -> None:
    """The token counter should work as a callable."""
    assert token_counter("Hello world") == token_counter.count("Hello world")

