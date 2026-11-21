"""Tests for the embedding exception hierarchy."""

import pytest

from app.embeddings.exceptions import (
    EmbeddingEncodingError,
    EmbeddingError,
    ModelLoadError,
)


def test_embedding_error_inherits_from_exception():
    assert issubclass(EmbeddingError, Exception)


def test_model_load_error_inherits_from_embedding_error():
    assert issubclass(ModelLoadError, EmbeddingError)


def test_encoding_error_inherits_from_embedding_error():
    assert issubclass(EmbeddingEncodingError, EmbeddingError)


@pytest.mark.parametrize(
    "exception_class",
    [
        EmbeddingError,
        ModelLoadError,
        EmbeddingEncodingError,
    ],
)
def test_custom_exceptions_preserve_error_message(exception_class):
    message = "Test embedding failure."

    error = exception_class(message)

    assert str(error) == message


def test_specific_exceptions_can_be_caught_as_embedding_error():
    with pytest.raises(EmbeddingError):
        raise ModelLoadError("Model initialization failed.")

    with pytest.raises(EmbeddingError):
        raise EmbeddingEncodingError("Encoding operation failed.")
