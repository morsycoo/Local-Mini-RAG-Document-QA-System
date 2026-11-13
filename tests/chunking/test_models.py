import pytest
from pydantic import ValidationError

from app.chunking.models import ChunkRecord


@pytest.fixture
def valid_chunk_data():
    content = "Machine Learning is interesting."

    return {
        "chunk_id": "chunk_001",
        "document_id": "document_001",
        "page_number": 1,
        "chunk_index": 0,
        "content": content,
        "char_start": 0,
        "char_end": len(content),
        "content_hash": "a" * 64,
        "section_path": ["Machine Learning"],
        "strategy": "fixed_window",
        "config_signature": "config_v1",
        "chunking_version": 1,
        "metadata": {},
    }


def test_valid_chunk_record(valid_chunk_data):
    chunk = ChunkRecord(**valid_chunk_data)

    assert chunk.chunk_id == "chunk_001"
    assert chunk.content == valid_chunk_data["content"]


def test_rejects_empty_chunk_id(valid_chunk_data):
    valid_chunk_data["chunk_id"] = ""

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_invalid_page_number(valid_chunk_data):
    valid_chunk_data["page_number"] = 0

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_negative_chunk_index(valid_chunk_data):
    valid_chunk_data["chunk_index"] = -1

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_invalid_content_hash(valid_chunk_data):
    valid_chunk_data["content_hash"] = "invalid"

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_inconsistent_offsets(valid_chunk_data):
    valid_chunk_data["char_end"] += 1

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_reversed_offsets(valid_chunk_data):
    valid_chunk_data["char_start"] = 10
    valid_chunk_data["char_end"] = 5

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_rejects_extra_fields(valid_chunk_data):
    valid_chunk_data["unknown_field"] = "unexpected"

    with pytest.raises(ValidationError):
        ChunkRecord(**valid_chunk_data)


def test_mutable_defaults_are_independent(valid_chunk_data):
    first_chunk = ChunkRecord(**valid_chunk_data)
    second_chunk = ChunkRecord(**valid_chunk_data)

    first_chunk.metadata["source"] = "test"

    assert "source" not in second_chunk.metadata
