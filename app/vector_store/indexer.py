"""Artifact loading and batch preparation for ChromaDB indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.config import EmbeddingConfig
from app.vector_store.config import VectorStoreConfig
from app.vector_store.exceptions import VectorStoreIntegrityError
from app.vector_store.metadata import map_chunk_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings" / "embeddings.npz"
MANIFEST_PATH = PROCESSED_DIR / "embeddings" / "manifest.json"


def _sha256_file(path: Path) -> str:
    """Calculate the SHA256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_and_validate_artifacts(
    embedding_config: EmbeddingConfig | None = None,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    list[str],
]:
    """Load and validate chunks, embeddings, and their manifest."""

    embedding_config = embedding_config or EmbeddingConfig()

    for path in (
        CHUNKS_PATH,
        EMBEDDINGS_PATH,
        MANIFEST_PATH,
    ):
        if not path.is_file():
            raise VectorStoreIntegrityError(f"Required artifact does not exist: {path}")

    try:
        with CHUNKS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks = [json.loads(line) for line in file if line.strip()]

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        with np.load(
            EMBEDDINGS_PATH,
            allow_pickle=False,
        ) as archive:
            if not {"embeddings", "chunk_ids"}.issubset(archive.files):
                raise VectorStoreIntegrityError(
                    "Embedding archive is missing required arrays."
                )

            embeddings = archive["embeddings"].copy()
            embedding_ids = archive["chunk_ids"].tolist()

    except VectorStoreIntegrityError:
        raise

    except Exception as error:
        raise VectorStoreIntegrityError("Failed to load indexing artifacts.") from error

    if not chunks:
        raise VectorStoreIntegrityError("The processed corpus is empty.")

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    if not all(isinstance(value, str) and value.strip() for value in chunk_ids):
        raise VectorStoreIntegrityError("Corpus contains missing or invalid chunk IDs.")

    if len(set(chunk_ids)) != len(chunk_ids):
        raise VectorStoreIntegrityError("Corpus contains duplicate chunk IDs.")

    if len(set(embedding_ids)) != len(embedding_ids):
        raise VectorStoreIntegrityError(
            "Embedding archive contains duplicate chunk IDs."
        )

    if chunk_ids != embedding_ids:
        raise VectorStoreIntegrityError("Chunk IDs and embedding IDs are not aligned.")

    expected_shape = (
        len(chunks),
        embedding_config.embedding_dimension,
    )

    if embeddings.shape != expected_shape:
        raise VectorStoreIntegrityError(
            f"Invalid embedding shape: expected {expected_shape}, "
            f"got {embeddings.shape}."
        )

    if embeddings.dtype != np.float32:
        raise VectorStoreIntegrityError("Embedding array must use float32.")

    if not np.isfinite(embeddings).all():
        raise VectorStoreIntegrityError("Embedding array contains non-finite values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise VectorStoreIntegrityError("Embedding vectors are not normalized.")

    source_hash = _sha256_file(CHUNKS_PATH)
    embedding_hash = _sha256_file(EMBEDDINGS_PATH)

    if source_hash != manifest["source"]["sha256"]:
        raise VectorStoreIntegrityError(
            "Source corpus hash does not match the manifest."
        )

    if embedding_hash != manifest["embeddings"]["sha256"]:
        raise VectorStoreIntegrityError(
            "Embedding artifact hash does not match the manifest."
        )

    if manifest["source"]["chunk_count"] != len(chunks):
        raise VectorStoreIntegrityError("Manifest source chunk count mismatch.")

    if manifest["embeddings"]["count"] != len(embeddings):
        raise VectorStoreIntegrityError("Manifest embedding count mismatch.")

    if manifest["embeddings"]["dimension"] != (embedding_config.embedding_dimension):
        raise VectorStoreIntegrityError("Manifest embedding dimension mismatch.")

    if manifest["embeddings"]["dtype"] != "float32":
        raise VectorStoreIntegrityError("Manifest embedding dtype mismatch.")

    if manifest["embeddings"]["normalized"] is not True:
        raise VectorStoreIntegrityError(
            "Manifest does not identify normalized embeddings."
        )

    if manifest["model"]["model_id"] != embedding_config.model_id:
        raise VectorStoreIntegrityError("Manifest embedding model mismatch.")

    if manifest["model"]["revision"] != embedding_config.revision:
        raise VectorStoreIntegrityError("Manifest model revision mismatch.")

    if manifest["model"]["embedding_dimension"] != (
        embedding_config.embedding_dimension
    ):
        raise VectorStoreIntegrityError("Manifest model dimension mismatch.")

    return chunks, embeddings, chunk_ids


def prepare_index_batches(
    batch_size: int = 64,
    embedding_config: EmbeddingConfig | None = None,
) -> list[dict[str, list[Any]]]:
    """Prepare validated ChromaDB batches without writing to the database."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    embedding_config = embedding_config or EmbeddingConfig()

    chunks, embeddings, chunk_ids = load_and_validate_artifacts(embedding_config)

    batches: list[dict[str, list[Any]]] = []

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        batch_chunks = chunks[start:end]

        batch = {
            "ids": chunk_ids[start:end],
            "documents": [chunk["content"] for chunk in batch_chunks],
            "embeddings": embeddings[start:end].tolist(),
            "metadatas": [
                map_chunk_metadata(
                    chunk,
                    embedding_config,
                )
                for chunk in batch_chunks
            ],
        }

        batches.append(batch)

    return batches


def index_chunks(
    batch_size: int = 64,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Index validated chunks into the persistent ChromaDB collection."""

    from app.vector_store.chroma_store import ChromaStore

    embedding_config = embedding_config or EmbeddingConfig()

    store = ChromaStore(
        embedding_config=embedding_config,
    )

    collection = store.get_or_create_collection()

    # Validate collection configuration before writing.
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise VectorStoreIntegrityError("Collection must use cosine distance.")

    if collection.metadata.get("embedding_model") != (embedding_config.model_id):
        raise VectorStoreIntegrityError("Collection embedding model mismatch.")

    if collection.metadata.get("embedding_revision") != (embedding_config.revision):
        raise VectorStoreIntegrityError("Collection embedding revision mismatch.")

    if collection.metadata.get("embedding_dimension") != (
        embedding_config.embedding_dimension
    ):
        raise VectorStoreIntegrityError("Collection embedding dimension mismatch.")

    chunks, embeddings, chunk_ids = load_and_validate_artifacts(embedding_config)

    expected_ids = set(chunk_ids)

    # Prepare all batches before the first database write.
    batches = prepare_index_batches(
        batch_size=batch_size,
        embedding_config=embedding_config,
    )

    # Reject records that do not belong to the current corpus.
    existing_records = collection.get(include=[])
    existing_ids = set(existing_records["ids"])

    unexpected_ids = existing_ids - expected_ids

    if unexpected_ids:
        raise VectorStoreIntegrityError(
            "Collection contains IDs outside the current corpus."
        )

    batch_results: list[dict[str, int]] = []

    for batch_number, batch in enumerate(batches, start=1):
        # Recheck collection state before each write.
        current_records = collection.get(include=[])
        current_ids = set(current_records["ids"])

        unexpected_ids = current_ids - expected_ids

        if unexpected_ids:
            raise VectorStoreIntegrityError(
                "Unexpected records detected before batch upsert."
            )

        collection.upsert(
            ids=batch["ids"],
            documents=batch["documents"],
            embeddings=batch["embeddings"],
            metadatas=batch["metadatas"],
        )

        # Verify that every ID in this batch was persisted.
        persisted_records = collection.get(
            ids=batch["ids"],
            include=[],
        )

        persisted_ids = set(persisted_records["ids"])
        expected_batch_ids = set(batch["ids"])

        if persisted_ids != expected_batch_ids:
            raise VectorStoreIntegrityError(
                f"Batch {batch_number} persistence verification failed."
            )

        # Verify collection count after this batch.
        expected_count = len(current_ids | expected_batch_ids)
        actual_count = collection.count()

        if actual_count != expected_count:
            raise VectorStoreIntegrityError(
                f"Batch {batch_number} count mismatch: "
                f"expected {expected_count}, got {actual_count}."
            )

        batch_results.append(
            {
                "batch_number": batch_number,
                "records_upserted": len(batch["ids"]),
                "collection_count": actual_count,
            }
        )

    # Final collection integrity verification.
    final_records = collection.get(include=[])
    final_ids = set(final_records["ids"])

    if final_ids != expected_ids:
        missing_ids = expected_ids - final_ids
        unexpected_ids = final_ids - expected_ids

        raise VectorStoreIntegrityError(
            "Final indexing verification failed. "
            f"Missing IDs: {len(missing_ids)}; "
            f"unexpected IDs: {len(unexpected_ids)}."
        )

    if collection.count() != len(chunks):
        raise VectorStoreIntegrityError(
            "Final collection count does not match corpus count."
        )

    return {
        "collection_name": collection.name,
        "total_chunks": len(chunks),
        "total_vectors": len(embeddings),
        "total_batches": len(batches),
        "batch_results": batch_results,
        "final_count": collection.count(),
        "unique_ids_verified": len(final_ids),
        "integrity_check": "passed",
    }
