"""Tests for ChromaDB client and collection management."""

from unittest.mock import Mock, patch

import pytest

from app.embeddings.config import EmbeddingConfig
from app.vector_store.chroma_store import ChromaStore
from app.vector_store.config import VectorStoreConfig
from app.vector_store.exceptions import (
    CollectionConfigurationError,
    VectorStoreConnectionError,
)


@pytest.fixture
def store_config(tmp_path):
    return VectorStoreConfig(
        persist_directory=tmp_path / "chroma",
    )


@pytest.fixture
def store(store_config):
    return ChromaStore(config=store_config)


def test_client_property_requires_connection(store):
    with pytest.raises(VectorStoreConnectionError):
        _ = store.client


def test_collection_property_requires_initialization(store):
    with pytest.raises(CollectionConfigurationError):
        _ = store.collection


def test_connect_creates_persistent_directory(store):
    client = store.connect()

    assert client is not None
    assert store.config.persist_directory.is_dir()
    assert store.client is client


def test_connect_reuses_existing_client(store):
    first_client = store.connect()
    second_client = store.connect()

    assert first_client is second_client


def test_connect_wraps_initialization_errors(store):
    with patch(
        "app.vector_store.chroma_store.chromadb.PersistentClient",
        side_effect=RuntimeError("Connection failed"),
    ):
        with pytest.raises(VectorStoreConnectionError):
            store.connect()


def test_expected_collection_metadata(store):
    metadata = store._expected_collection_metadata()

    assert metadata == {
        "embedding_model": store.embedding_config.model_id,
        "embedding_revision": store.embedding_config.revision,
        "embedding_dimension": 768,
        "distance_metric": "cosine",
    }


def test_create_collection_with_expected_configuration(store):
    collection = store.get_or_create_collection()

    assert collection.name == store.config.collection_name
    assert collection.metadata == store._expected_collection_metadata()
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.count() == 0
    assert store.collection is collection


def test_collection_uses_external_embeddings(store):
    collection = store.get_or_create_collection()

    assert collection._embedding_function is None


def test_get_or_create_collection_reuses_collection(store):
    first_collection = store.get_or_create_collection()
    second_collection = store.get_or_create_collection()

    assert first_collection.name == second_collection.name
    assert first_collection.id == second_collection.id


def test_dimension_mismatch_is_rejected(store_config):
    invalid_embedding_config = EmbeddingConfig(
        embedding_dimension=384,
    )

    with pytest.raises(CollectionConfigurationError):
        ChromaStore(
            config=store_config,
            embedding_config=invalid_embedding_config,
        )


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("embedding_model", "different-model"),
        ("embedding_revision", "different-revision"),
        ("embedding_dimension", 384),
        ("distance_metric", "l2"),
    ],
)
def test_collection_metadata_mismatch_is_rejected(
    store_config,
    metadata_key,
    metadata_value,
):
    store = ChromaStore(config=store_config)
    client = store.connect()

    metadata = store._expected_collection_metadata()
    metadata[metadata_key] = metadata_value

    collection = client.create_collection(
        name="metadata_probe",
        metadata=metadata,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )

    with pytest.raises(CollectionConfigurationError):
        store._validate_collection(collection)


def test_collection_distance_metric_mismatch_is_rejected(
    store_config,
):
    store = ChromaStore(config=store_config)
    client = store.connect()

    collection = client.create_collection(
        name="distance_probe",
        metadata=store._expected_collection_metadata(),
        configuration={"hnsw": {"space": "l2"}},
        embedding_function=None,
    )

    with pytest.raises(CollectionConfigurationError):
        store._validate_collection(collection)


def test_collection_creation_errors_are_wrapped(store):
    store.connect()

    with patch.object(
        store.client,
        "get_or_create_collection",
        side_effect=RuntimeError("Collection creation failed"),
    ):
        with pytest.raises(CollectionConfigurationError):
            store.get_or_create_collection()


def test_connection_error_is_not_reclassified(store):
    with patch.object(
        store,
        "connect",
        side_effect=VectorStoreConnectionError("Connection failed"),
    ):
        with pytest.raises(VectorStoreConnectionError):
            store.get_or_create_collection()
