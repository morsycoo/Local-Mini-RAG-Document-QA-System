"""Tests for vector store exceptions."""

import pytest

from app.vector_store.exceptions import (
    CollectionConfigurationError,
    EmbeddingDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreIntegrityError,
)


@pytest.mark.parametrize(
    "exception_class",
    [
        VectorStoreConnectionError,
        CollectionConfigurationError,
        EmbeddingDimensionMismatchError,
        VectorStoreIntegrityError,
    ],
)
def test_custom_exceptions_inherit_from_vector_store_error(
    exception_class: type[Exception],
):
    assert issubclass(exception_class, VectorStoreError)


def test_vector_store_error_inherits_from_exception():
    assert issubclass(VectorStoreError, Exception)


@pytest.mark.parametrize(
    "exception_class",
    [
        VectorStoreError,
        VectorStoreConnectionError,
        CollectionConfigurationError,
        EmbeddingDimensionMismatchError,
        VectorStoreIntegrityError,
    ],
)
def test_custom_exceptions_preserve_error_message(
    exception_class: type[Exception],
):
    message = "Vector store test error."

    error = exception_class(message)

    assert str(error) == message


def test_specific_exception_can_be_caught_as_base_error():
    with pytest.raises(VectorStoreError):
        raise VectorStoreConnectionError("Database connection failed.")
