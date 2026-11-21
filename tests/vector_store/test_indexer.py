"""Tests for ChromaDB artifact loading and batch preparation."""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from app.embeddings.config import EmbeddingConfig
from app.vector_store.exceptions import VectorStoreIntegrityError
from app.vector_store.indexer import (
    load_and_validate_artifacts,
    prepare_index_batches,
)


# ============================================================
# 1. Fixtures
# ============================================================


@pytest.fixture
def sample_artifacts():
    """Create in-memory chunks and normalized embeddings."""

    chunks = [
        {
            "chunk_id": f"chunk_{index:03d}",
            "document_id": "doc_001",
            "page_number": 1,
            "chunk_index": index,
            "content": f"Test content {index}",
            "char_start": index * 20,
            "char_end": index * 20 + 15,
            "content_hash": f"{index:064x}",
            "section_path": ["Introduction"],
            "strategy": "recursive",
            "config_signature": "config_v1",
            "chunking_version": 1,
            "metadata": {
                "file_name": "document.pdf",
                "source_path": "data/document.pdf",
                "file_type": "pdf",
                "block_type": "paragraph",
                "chunk_size_chars": 1000,
                "min_chunk_size_chars": 100,
            },
        }
        for index in range(10)
    ]

    embeddings = np.zeros(
        (10, 768),
        dtype=np.float32,
    )
    embeddings[:, 0] = 1.0

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    return chunks, embeddings, chunk_ids


@pytest.fixture
def artifact_files(tmp_path, monkeypatch):
    """Create isolated and valid indexing artifacts."""

    config = EmbeddingConfig()

    chunk = {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "page_number": 1,
        "chunk_index": 0,
        "content": "A test document chunk.",
        "char_start": 0,
        "char_end": 23,
        "content_hash": "a" * 64,
        "section_path": ["Introduction"],
        "strategy": "recursive",
        "config_signature": "config_v1",
        "chunking_version": 1,
        "metadata": {
            "file_name": "document.pdf",
            "source_path": "data/document.pdf",
            "file_type": "pdf",
            "block_type": "paragraph",
            "chunk_size_chars": 1000,
            "min_chunk_size_chars": 100,
        },
    }

    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npz"
    manifest_path = tmp_path / "manifest.json"

    # Save processed chunks.
    chunks_path.write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Create a normalized embedding.
    embeddings = np.zeros(
        (1, config.embedding_dimension),
        dtype=np.float32,
    )
    embeddings[0, 0] = 1.0

    # Save embeddings and their corresponding IDs.
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        chunk_ids=np.array(["chunk_001"]),
    )

    def calculate_sha256(path):
        """Calculate SHA256 for a file."""

        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Build a manifest matching the generated artifacts.
    manifest = {
        "source": {
            "sha256": calculate_sha256(chunks_path),
            "chunk_count": 1,
        },
        "embeddings": {
            "sha256": calculate_sha256(embeddings_path),
            "count": 1,
            "dimension": config.embedding_dimension,
            "dtype": "float32",
            "normalized": True,
        },
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "embedding_dimension": config.embedding_dimension,
        },
    }

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    # Redirect production artifact paths to temporary files.
    monkeypatch.setattr(
        "app.vector_store.indexer.CHUNKS_PATH",
        chunks_path,
    )
    monkeypatch.setattr(
        "app.vector_store.indexer.EMBEDDINGS_PATH",
        embeddings_path,
    )
    monkeypatch.setattr(
        "app.vector_store.indexer.MANIFEST_PATH",
        manifest_path,
    )

    return {
        "chunks_path": chunks_path,
        "embeddings_path": embeddings_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "chunk": chunk,
        "embeddings": embeddings,
        "config": config,
    }


# ============================================================
# 2. Batch Preparation Tests
# ============================================================


def test_prepare_batches_returns_expected_batch_count(
    sample_artifacts,
):
    """Verify that chunks are divided into the expected batches."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=4)

    assert len(batches) == 3
    assert [len(batch["ids"]) for batch in batches] == [4, 4, 2]


def test_prepare_batches_preserves_chunk_id_order(
    sample_artifacts,
):
    """Verify that chunk ID ordering remains unchanged."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=3)

    actual_ids = [chunk_id for batch in batches for chunk_id in batch["ids"]]

    assert actual_ids == chunk_ids


def test_prepare_batches_preserves_document_content(
    sample_artifacts,
):
    """Verify that document content is preserved."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=4)

    documents = [document for batch in batches for document in batch["documents"]]

    assert documents == [chunk["content"] for chunk in chunks]


def test_prepare_batches_preserves_embedding_alignment(
    sample_artifacts,
):
    """Verify embedding order and dimensions across batches."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=4)

    actual_embeddings = np.asarray(
        [vector for batch in batches for vector in batch["embeddings"]],
        dtype=np.float32,
    )

    assert actual_embeddings.shape == (10, 768)

    np.testing.assert_array_equal(
        actual_embeddings,
        embeddings,
    )


def test_prepare_batches_maps_metadata_for_every_chunk(
    sample_artifacts,
):
    """Verify that every chunk receives mapped metadata."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=4)

    metadatas = [metadata for batch in batches for metadata in batch["metadatas"]]

    assert len(metadatas) == len(chunks)
    assert all(len(metadata) == 19 for metadata in metadatas)


def test_prepare_batches_keeps_all_components_aligned(
    sample_artifacts,
):
    """Verify equal component counts within every batch."""

    chunks, embeddings, chunk_ids = sample_artifacts

    with patch(
        "app.vector_store.indexer.load_and_validate_artifacts",
        return_value=(chunks, embeddings, chunk_ids),
    ):
        batches = prepare_index_batches(batch_size=4)

    for batch in batches:
        lengths = {
            len(batch["ids"]),
            len(batch["documents"]),
            len(batch["embeddings"]),
            len(batch["metadatas"]),
        }

        assert len(lengths) == 1


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        1.5,
        True,
        "4",
        None,
    ],
)
def test_prepare_batches_rejects_invalid_batch_size(batch_size):
    """Reject invalid batch sizes."""

    with pytest.raises(ValueError):
        prepare_index_batches(
            batch_size=batch_size,
        )


# ============================================================
# 3. Artifact Loading Tests
# ============================================================


def test_load_valid_artifacts(artifact_files):
    """Verify successful loading of valid artifacts."""

    chunks, embeddings, chunk_ids = load_and_validate_artifacts()

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk_001"

    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32

    assert chunk_ids == ["chunk_001"]


def test_load_artifacts_preserves_embedding_alignment(
    artifact_files,
):
    """Verify that loaded embeddings remain aligned with chunk IDs."""

    chunks, embeddings, chunk_ids = load_and_validate_artifacts()

    assert chunk_ids == [chunk["chunk_id"] for chunk in chunks]

    np.testing.assert_array_equal(
        embeddings,
        artifact_files["embeddings"],
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "chunks_path",
        "embeddings_path",
        "manifest_path",
    ],
)
def test_load_artifacts_rejects_missing_files(
    artifact_files,
    artifact_name,
):
    """Reject missing indexing artifacts."""

    path = artifact_files[artifact_name]
    path.unlink()

    with pytest.raises(VectorStoreIntegrityError):
        load_and_validate_artifacts()


# ============================================================
# 4. Artifact Integrity Tests
# ============================================================


def test_load_artifacts_rejects_source_hash_mismatch(
    artifact_files,
):
    """Reject a source corpus that does not match its manifest hash."""

    path = artifact_files["chunks_path"]

    path.write_text(
        path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="Source corpus hash does not match the manifest",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_embedding_hash_mismatch(
    artifact_files,
):
    """Reject an embedding archive that does not match its manifest hash."""

    path = artifact_files["embeddings_path"]

    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(
        VectorStoreIntegrityError,
        match="Embedding artifact hash does not match the manifest",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_empty_corpus(
    artifact_files,
):
    """Reject an empty processed corpus."""

    artifact_files["chunks_path"].write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="processed corpus is empty",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_duplicate_chunk_ids(
    artifact_files,
):
    """Reject duplicate chunk IDs in the source corpus."""

    chunk = artifact_files["chunk"]

    duplicate_chunks = [
        chunk,
        chunk,
    ]

    artifact_files["chunks_path"].write_text(
        "".join(json.dumps(item) + "\n" for item in duplicate_chunks),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="duplicate chunk IDs",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_embedding_id_mismatch(
    artifact_files,
):
    """Reject embeddings whose IDs do not match the source corpus."""

    np.savez_compressed(
        artifact_files["embeddings_path"],
        embeddings=artifact_files["embeddings"],
        chunk_ids=np.array(["wrong_id"]),
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="not aligned",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_wrong_embedding_shape(
    artifact_files,
):
    """Reject embeddings with an incorrect dimension."""

    np.savez_compressed(
        artifact_files["embeddings_path"],
        embeddings=np.ones(
            (1, 384),
            dtype=np.float32,
        ),
        chunk_ids=np.array(["chunk_001"]),
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="Invalid embedding shape",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_non_float32_embeddings(
    artifact_files,
):
    """Reject embeddings that are not float32."""

    embeddings = np.zeros(
        (1, 768),
        dtype=np.float64,
    )
    embeddings[0, 0] = 1.0

    np.savez_compressed(
        artifact_files["embeddings_path"],
        embeddings=embeddings,
        chunk_ids=np.array(["chunk_001"]),
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="float32",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_non_finite_embeddings(
    artifact_files,
):
    """Reject embeddings containing NaN values."""

    embeddings = artifact_files["embeddings"].copy()
    embeddings[0, 0] = np.nan

    np.savez_compressed(
        artifact_files["embeddings_path"],
        embeddings=embeddings,
        chunk_ids=np.array(["chunk_001"]),
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="non-finite",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_unnormalized_embeddings(
    artifact_files,
):
    """Reject embeddings that are not normalized."""

    embeddings = np.zeros(
        (1, 768),
        dtype=np.float32,
    )
    embeddings[0, 0] = 2.0

    np.savez_compressed(
        artifact_files["embeddings_path"],
        embeddings=embeddings,
        chunk_ids=np.array(["chunk_001"]),
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="not normalized",
    ):
        load_and_validate_artifacts()


# ============================================================
# 5. Manifest Validation Tests
# ============================================================


def test_load_artifacts_rejects_manifest_chunk_count_mismatch(
    artifact_files,
):
    """Reject a manifest with an incorrect source chunk count."""

    manifest = artifact_files["manifest"]
    manifest["source"]["chunk_count"] = 99

    artifact_files["manifest_path"].write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="chunk count mismatch",
    ):
        load_and_validate_artifacts()


def test_load_artifacts_rejects_manifest_model_mismatch(
    artifact_files,
):
    """Reject a manifest with an incompatible embedding model."""

    manifest = artifact_files["manifest"]
    manifest["model"]["model_id"] = "different-model"

    artifact_files["manifest_path"].write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="model mismatch",
    ):
        load_and_validate_artifacts()


# ============================================================
# 6. Persistent Indexing Tests
# ============================================================



@pytest.fixture
def isolated_indexing_environment(
    artifact_files,
    tmp_path,
    monkeypatch,
):
    """Create an isolated ChromaDB environment with one shared collection."""

    from app.vector_store.chroma_store import ChromaStore
    from app.vector_store.config import VectorStoreConfig

    config = artifact_files["config"]

    store_config = VectorStoreConfig(
        persist_directory=tmp_path / "chroma",
    )

    store = ChromaStore(
        config=store_config,
        embedding_config=config,
    )

    collection = store.get_or_create_collection()

    # Ensure every indexing call uses the exact same collection object.
    monkeypatch.setattr(
        store,
        "get_or_create_collection",
        lambda: collection,
    )

    monkeypatch.setattr(
        "app.vector_store.chroma_store.ChromaStore",
        lambda embedding_config=None: store,
    )

    return store


def test_index_chunks_indexes_all_records(
    isolated_indexing_environment,
):
    """Verify that all source chunks are indexed successfully."""

    from app.vector_store.indexer import index_chunks

    result = index_chunks(batch_size=1)

    collection = isolated_indexing_environment.get_or_create_collection()

    assert result["collection_name"] == collection.name
    assert result["total_chunks"] == 1
    assert result["total_vectors"] == 1
    assert result["total_batches"] == 1
    assert result["final_count"] == 1
    assert result["unique_ids_verified"] == 1
    assert result["integrity_check"] == "passed"

    assert collection.count() == 1


def test_index_chunks_returns_batch_results(
    isolated_indexing_environment,
):
    """Verify the indexing batch summary."""

    from app.vector_store.indexer import index_chunks

    result = index_chunks(batch_size=1)

    assert result["batch_results"] == [
        {
            "batch_number": 1,
            "records_upserted": 1,
            "collection_count": 1,
        }
    ]


def test_index_chunks_is_idempotent(
    isolated_indexing_environment,
):
    """Verify repeated indexing does not create duplicate records."""

    from app.vector_store.indexer import index_chunks

    first_result = index_chunks(batch_size=1)
    second_result = index_chunks(batch_size=1)

    collection = isolated_indexing_environment.get_or_create_collection()

    assert first_result["final_count"] == 1
    assert second_result["final_count"] == 1
    assert collection.count() == 1

    records = collection.get(include=[])

    assert records["ids"] == ["chunk_001"]


def test_index_chunks_rejects_unexpected_existing_ids(
    isolated_indexing_environment,
):
    """Reject records that are not part of the current corpus."""

    from app.vector_store.indexer import index_chunks

    collection = isolated_indexing_environment.get_or_create_collection()

    collection.upsert(
        ids=["unexpected_chunk"],
        documents=["Unexpected content"],
        embeddings=[[1.0] + [0.0] * 767],
        metadatas=[{"document_id": "unexpected_doc"}],
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="outside the current corpus",
    ):
        index_chunks(batch_size=1)

    assert collection.count() == 1


def test_index_chunks_rejects_collection_metadata_mismatch(
    isolated_indexing_environment,
):
    """Reject a collection with incompatible embedding metadata."""

    from app.vector_store.indexer import index_chunks

    collection = isolated_indexing_environment.get_or_create_collection()

    collection.metadata["embedding_model"] = "different-model"

    with pytest.raises(
        VectorStoreIntegrityError,
        match="embedding model mismatch",
    ):
        index_chunks(batch_size=1)


def test_index_chunks_rejects_collection_distance_mismatch(
    tmp_path,
    artifact_files,
    monkeypatch,
):
    """Reject a collection configured with a non-cosine distance metric."""

    from app.vector_store.chroma_store import ChromaStore
    from app.vector_store.config import VectorStoreConfig
    from app.vector_store.indexer import index_chunks

    config = artifact_files["config"]

    store = ChromaStore(
        config=VectorStoreConfig(
            persist_directory=tmp_path / "wrong_metric",
        ),
        embedding_config=config,
    )

    store.connect()

    collection = store.client.create_collection(
        name="hadota_chunks_e5_base_v1",
        metadata={
            "embedding_model": config.model_id,
            "embedding_revision": config.revision,
            "embedding_dimension": config.embedding_dimension,
            "distance_metric": "l2",
        },
        configuration={
            "hnsw": {
                "space": "l2",
            }
        },
        embedding_function=None,
    )

    monkeypatch.setattr(
        "app.vector_store.chroma_store.ChromaStore",
        lambda embedding_config=None: store,
    )

    with pytest.raises(
        Exception,
    ):
        index_chunks(batch_size=1)

    assert collection.count() == 0


def test_index_chunks_rejects_missing_persisted_records(
    isolated_indexing_environment,
    monkeypatch,
):
    """Verify that missing records after upsert are detected."""

    from app.vector_store.indexer import index_chunks

    collection = isolated_indexing_environment.get_or_create_collection()

    original_get = collection.get

    def get_without_persisted_ids(*args, **kwargs):
        result = original_get(*args, **kwargs)

        if kwargs.get("ids") == ["chunk_001"]:
            return {"ids": []}

        return result

    monkeypatch.setattr(
        collection,
        "get",
        get_without_persisted_ids,
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="persistence verification failed",
    ):
        index_chunks(batch_size=1)


def test_index_chunks_rejects_collection_count_mismatch(
    isolated_indexing_environment,
    monkeypatch,
):
    """Verify that incorrect collection counts are detected."""

    from app.vector_store.indexer import index_chunks

    collection = isolated_indexing_environment.get_or_create_collection()

    monkeypatch.setattr(
        collection,
        "count",
        lambda: 99,
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="count mismatch",
    ):
        index_chunks(batch_size=1)


def test_index_chunks_rejects_final_missing_ids(
    isolated_indexing_environment,
    monkeypatch,
):
    """Verify that missing IDs during final validation are detected."""

    from app.vector_store.indexer import index_chunks

    collection = isolated_indexing_environment.get_or_create_collection()

    original_get = collection.get

    def get_without_final_ids(*args, **kwargs):
        result = original_get(*args, **kwargs)

        if "ids" not in kwargs:
            return {"ids": []}

        return result

    monkeypatch.setattr(
        collection,
        "get",
        get_without_final_ids,
    )

    with pytest.raises(
        VectorStoreIntegrityError,
        match="Final indexing verification failed",
    ):
        index_chunks(batch_size=1)
