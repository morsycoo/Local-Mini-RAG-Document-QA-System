"""Tests for passage text preparation."""

import pytest

from app.embeddings.config import EmbeddingConfig
from app.embeddings.text import prepare_passage_text


@pytest.fixture
def config():
    return EmbeddingConfig()


def test_adds_passage_prefix(config):
    text = "Machine learning is useful."

    result = prepare_passage_text(text, config)

    assert result == "passage: Machine learning is useful."


def test_does_not_duplicate_existing_prefix(config):
    text = "passage: Machine learning is useful."

    result = prepare_passage_text(text, config)

    assert result == text


def test_preserves_original_whitespace(config):
    text = "  Machine learning is useful.  "

    result = prepare_passage_text(text, config)

    assert result == "passage:   Machine learning is useful.  "
    assert text == "  Machine learning is useful.  "


@pytest.mark.parametrize("text", ["", " ", "\n", "\t  "])
def test_rejects_empty_or_whitespace_only_text(text, config):
    with pytest.raises(ValueError):
        prepare_passage_text(text, config)


@pytest.mark.parametrize("text", [None, 123, 1.5, [], {}])
def test_rejects_non_string_input(text, config):
    with pytest.raises(TypeError):
        prepare_passage_text(text, config)


def test_uses_custom_passage_prefix():
    config = EmbeddingConfig(passage_prefix="document: ")

    result = prepare_passage_text("Example content.", config)

    assert result == "document: Example content."
