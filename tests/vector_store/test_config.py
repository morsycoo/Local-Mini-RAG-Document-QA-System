"""Tests for vector store configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.vector_store.config import (
    PROJECT_ROOT,
    VectorStoreConfig,
)


def test_default_configuration():
    config = VectorStoreConfig()

    assert config.collection_name == "hadota_chunks_e5_base_v1"
    assert config.distance_metric == "cosine"
    assert config.embedding_dimension == 768
    assert config.anonymized_telemetry is False


def test_default_persist_directory():
    config = VectorStoreConfig()

    expected_path = PROJECT_ROOT / "data" / "knowledge_base" / "vector_store" / "chroma"

    assert config.persist_directory == expected_path
    assert config.persist_directory.is_absolute()


def test_custom_persist_directory(tmp_path: Path):
    config = VectorStoreConfig(
        persist_directory=tmp_path,
    )

    assert config.persist_directory == tmp_path


def test_custom_collection_name():
    config = VectorStoreConfig(
        collection_name="custom_collection_v1",
    )

    assert config.collection_name == "custom_collection_v1"


@pytest.mark.parametrize(
    "collection_name",
    [
        "",
        " ",
        " collection",
        "collection ",
        "_collection",
        "-collection",
        "collection_",
        "collection-",
        "collection name",
        "collection/name",
    ],
)
def test_invalid_collection_names_are_rejected(
    collection_name: str,
):
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            collection_name=collection_name,
        )


def test_invalid_distance_metric_is_rejected():
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            distance_metric="l2",
        )


@pytest.mark.parametrize(
    "dimension",
    [
        0,
        -1,
    ],
)
def test_invalid_embedding_dimensions_are_rejected(
    dimension: int,
):
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            embedding_dimension=dimension,
        )


def test_unknown_configuration_fields_are_rejected():
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            unknown_setting=True,
        )


def test_configuration_is_immutable():
    config = VectorStoreConfig()

    with pytest.raises(ValidationError):
        config.embedding_dimension = 384
