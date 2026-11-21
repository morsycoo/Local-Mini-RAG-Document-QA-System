"""Tests for strict ChromaDB metadata mapping."""

import json
from copy import deepcopy

import pytest

from app.embeddings.config import EmbeddingConfig
from app.vector_store.exceptions import VectorStoreIntegrityError
from app.vector_store.metadata import map_chunk_metadata


@pytest.fixture
def embedding_config():
    return EmbeddingConfig()


@pytest.fixture
def valid_chunk():
    return {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "page_number": 1,
        "chunk_index": 0,
        "content": "This is a test document chunk.",
        "char_start": 0,
        "char_end": 31,
        "content_hash": "a" * 64,
        "section_path": ["Introduction", "Overview"],
        "strategy": "recursive",
        "config_signature": "config_v1",
        "chunking_version": 1,
        "metadata": {
            "file_name": "document.pdf",
            "source_path": "data/documents/document.pdf",
            "file_type": "pdf",
            "block_type": "paragraph",
            "chunk_size_chars": 1000,
            "min_chunk_size_chars": 100,
        },
    }


def test_maps_valid_chunk_metadata(valid_chunk, embedding_config):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert len(result) == 19
    assert result["document_id"] == "doc_001"
    assert result["file_name"] == "document.pdf"
    assert result["page_number"] == 1
    assert result["chunk_index"] == 0


def test_excludes_chunk_id_and_content(valid_chunk, embedding_config):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert "chunk_id" not in result
    assert "content" not in result


def test_maps_all_required_metadata_fields(valid_chunk, embedding_config):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    expected_fields = {
        "document_id",
        "file_name",
        "source_path",
        "file_type",
        "block_type",
        "page_number",
        "chunk_index",
        "char_start",
        "char_end",
        "section_path",
        "strategy",
        "config_signature",
        "chunking_version",
        "content_hash",
        "chunk_size_chars",
        "min_chunk_size_chars",
        "embedding_model",
        "embedding_revision",
        "embedding_dimension",
    }

    assert set(result) == expected_fields


def test_section_path_is_serialized_as_json(valid_chunk, embedding_config):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert json.loads(result["section_path"]) == [
        "Introduction",
        "Overview",
    ]


def test_section_path_preserves_unicode(valid_chunk, embedding_config):
    valid_chunk["section_path"] = ["المقدمة", "نظرة عامة"]

    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert "المقدمة" in result["section_path"]
    assert json.loads(result["section_path"]) == [
        "المقدمة",
        "نظرة عامة",
    ]


def test_embedding_configuration_is_mapped(valid_chunk, embedding_config):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert result["embedding_model"] == embedding_config.model_id
    assert result["embedding_revision"] == embedding_config.revision
    assert result["embedding_dimension"] == 768


def test_mapped_metadata_uses_chromadb_compatible_types(
    valid_chunk,
    embedding_config,
):
    result = map_chunk_metadata(valid_chunk, embedding_config)

    assert all(isinstance(value, (str, int, float, bool)) for value in result.values())


def test_non_dictionary_chunk_is_rejected(embedding_config):
    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata([], embedding_config)


@pytest.mark.parametrize(
    "field",
    [
        "chunk_id",
        "document_id",
        "page_number",
        "chunk_index",
        "content",
        "char_start",
        "char_end",
        "content_hash",
        "section_path",
        "strategy",
        "config_signature",
        "chunking_version",
        "metadata",
    ],
)
def test_missing_root_fields_are_rejected(
    valid_chunk,
    embedding_config,
    field,
):
    chunk = deepcopy(valid_chunk)
    del chunk[field]

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


@pytest.mark.parametrize(
    "field",
    [
        "file_name",
        "source_path",
        "file_type",
        "block_type",
        "chunk_size_chars",
        "min_chunk_size_chars",
    ],
)
def test_missing_nested_metadata_fields_are_rejected(
    valid_chunk,
    embedding_config,
    field,
):
    chunk = deepcopy(valid_chunk)
    del chunk["metadata"][field]

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", ""),
        ("document_id", " "),
        ("content", ""),
        ("content_hash", ""),
        ("strategy", ""),
        ("config_signature", ""),
    ],
)
def test_empty_root_strings_are_rejected(
    valid_chunk,
    embedding_config,
    field,
    value,
):
    chunk = deepcopy(valid_chunk)
    chunk[field] = value

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_number", 0),
        ("page_number", -1),
        ("chunk_index", -1),
        ("char_start", -1),
        ("char_end", -1),
        ("chunking_version", 0),
    ],
)
def test_invalid_numeric_root_fields_are_rejected(
    valid_chunk,
    embedding_config,
    field,
    value,
):
    chunk = deepcopy(valid_chunk)
    chunk[field] = value

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


@pytest.mark.parametrize(
    "field",
    [
        "page_number",
        "chunk_index",
        "char_start",
        "char_end",
        "chunking_version",
    ],
)
def test_boolean_values_are_rejected_as_integers(
    valid_chunk,
    embedding_config,
    field,
):
    chunk = deepcopy(valid_chunk)
    chunk[field] = True

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


def test_reversed_character_offsets_are_rejected(
    valid_chunk,
    embedding_config,
):
    valid_chunk["char_start"] = 20
    valid_chunk["char_end"] = 10

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(valid_chunk, embedding_config)


def test_non_dictionary_nested_metadata_is_rejected(
    valid_chunk,
    embedding_config,
):
    valid_chunk["metadata"] = []

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(valid_chunk, embedding_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_name", ""),
        ("source_path", " "),
        ("file_type", ""),
        ("block_type", ""),
    ],
)
def test_empty_nested_strings_are_rejected(
    valid_chunk,
    embedding_config,
    field,
    value,
):
    chunk = deepcopy(valid_chunk)
    chunk["metadata"][field] = value

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size_chars", 0),
        ("chunk_size_chars", -1),
        ("min_chunk_size_chars", 0),
        ("min_chunk_size_chars", -1),
    ],
)
def test_invalid_chunk_size_values_are_rejected(
    valid_chunk,
    embedding_config,
    field,
    value,
):
    chunk = deepcopy(valid_chunk)
    chunk["metadata"][field] = value

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(chunk, embedding_config)


def test_boolean_chunk_size_is_rejected(
    valid_chunk,
    embedding_config,
):
    valid_chunk["metadata"]["chunk_size_chars"] = True

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(valid_chunk, embedding_config)


def test_minimum_chunk_size_cannot_exceed_maximum(
    valid_chunk,
    embedding_config,
):
    valid_chunk["metadata"]["min_chunk_size_chars"] = 1200

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(valid_chunk, embedding_config)


@pytest.mark.parametrize(
    "section_path",
    [
        [],
        "",
        ["Introduction", ""],
        ["Introduction", "  "],
        ["Introduction", 123],
        None,
    ],
)
def test_invalid_section_paths_are_rejected(
    valid_chunk,
    embedding_config,
    section_path,
):
    valid_chunk["section_path"] = section_path

    with pytest.raises(VectorStoreIntegrityError):
        map_chunk_metadata(valid_chunk, embedding_config)
