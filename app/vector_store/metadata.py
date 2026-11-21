"""Strict metadata mapping and validation for ChromaDB."""

from __future__ import annotations

import json
from typing import Any

from app.embeddings.config import EmbeddingConfig
from app.vector_store.exceptions import VectorStoreIntegrityError


def _require_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise VectorStoreIntegrityError(
            f"Field '{field_name}' must be a non-empty string."
        )

    return value


def _require_integer(
    value: Any,
    field_name: str,
    minimum: int = 0,
) -> int:
    """Validate an integer without accepting booleans."""

    if type(value) is not int:
        raise VectorStoreIntegrityError(f"Field '{field_name}' must be an integer.")

    if value < minimum:
        raise VectorStoreIntegrityError(f"Field '{field_name}' must be >= {minimum}.")

    return value


def map_chunk_metadata(
    chunk: dict[str, Any],
    embedding_config: EmbeddingConfig,
) -> dict[str, str | int | float | bool]:
    """Validate a processed chunk and map its metadata to ChromaDB."""

    if not isinstance(chunk, dict):
        raise VectorStoreIntegrityError("Chunk must be a dictionary.")

    required_root_fields = {
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
    }

    missing_fields = required_root_fields - chunk.keys()

    if missing_fields:
        raise VectorStoreIntegrityError(
            f"Chunk is missing required fields: {sorted(missing_fields)}"
        )

    # Validate chunk identity and content.
    _require_string(chunk["chunk_id"], "chunk_id")
    _require_string(chunk["document_id"], "document_id")
    _require_string(chunk["content"], "content")
    _require_string(chunk["content_hash"], "content_hash")
    _require_string(chunk["strategy"], "strategy")
    _require_string(chunk["config_signature"], "config_signature")

    # Validate numeric root fields.
    page_number = _require_integer(
        chunk["page_number"],
        "page_number",
        minimum=1,
    )

    chunk_index = _require_integer(
        chunk["chunk_index"],
        "chunk_index",
    )

    char_start = _require_integer(
        chunk["char_start"],
        "char_start",
    )

    char_end = _require_integer(
        chunk["char_end"],
        "char_end",
    )

    _require_integer(
        chunk["chunking_version"],
        "chunking_version",
        minimum=1,
    )

    if char_end < char_start:
        raise VectorStoreIntegrityError(
            "char_end must be greater than or equal to char_start."
        )

    # Validate nested metadata.
    source_metadata = chunk["metadata"]

    if not isinstance(source_metadata, dict):
        raise VectorStoreIntegrityError("Chunk metadata must be a dictionary.")

    required_metadata_fields = {
        "file_name",
        "source_path",
        "file_type",
        "block_type",
        "chunk_size_chars",
        "min_chunk_size_chars",
    }

    missing_metadata_fields = required_metadata_fields - source_metadata.keys()

    if missing_metadata_fields:
        raise VectorStoreIntegrityError(
            f"Chunk metadata is missing fields: {sorted(missing_metadata_fields)}"
        )

    file_name = _require_string(
        source_metadata["file_name"],
        "metadata.file_name",
    )

    source_path = _require_string(
        source_metadata["source_path"],
        "metadata.source_path",
    )

    file_type = _require_string(
        source_metadata["file_type"],
        "metadata.file_type",
    )

    block_type = _require_string(
        source_metadata["block_type"],
        "metadata.block_type",
    )

    chunk_size_chars = _require_integer(
        source_metadata["chunk_size_chars"],
        "metadata.chunk_size_chars",
        minimum=1,
    )

    min_chunk_size_chars = _require_integer(
        source_metadata["min_chunk_size_chars"],
        "metadata.min_chunk_size_chars",
        minimum=1,
    )

    if min_chunk_size_chars > chunk_size_chars:
        raise VectorStoreIntegrityError(
            "min_chunk_size_chars cannot exceed chunk_size_chars."
        )

    # Validate hierarchical section path.
    section_path = chunk["section_path"]

    if (
        not isinstance(section_path, list)
        or not section_path
        or not all(
            isinstance(section, str) and section.strip() for section in section_path
        )
    ):
        raise VectorStoreIntegrityError(
            "section_path must be a non-empty list of non-empty strings."
        )

    # Build ChromaDB-compatible metadata.
    mapped_metadata: dict[str, str | int | float | bool] = {
        "document_id": chunk["document_id"],
        "file_name": file_name,
        "source_path": source_path,
        "file_type": file_type,
        "block_type": block_type,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "char_start": char_start,
        "char_end": char_end,
        "section_path": json.dumps(
            section_path,
            ensure_ascii=False,
        ),
        "strategy": chunk["strategy"],
        "config_signature": chunk["config_signature"],
        "chunking_version": chunk["chunking_version"],
        "content_hash": chunk["content_hash"],
        "chunk_size_chars": chunk_size_chars,
        "min_chunk_size_chars": min_chunk_size_chars,
        "embedding_model": embedding_config.model_id,
        "embedding_revision": embedding_config.revision,
        "embedding_dimension": embedding_config.embedding_dimension,
    }

    return mapped_metadata
