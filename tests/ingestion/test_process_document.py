"""Integration tests for document processing."""

from unittest.mock import patch

import pytest

from app.ingestion import pipeline
from app.ingestion.discovery import DiscoveredFile
from app.ingestion.models import IngestionStatus
from app.ingestion.pipeline import (
    load_existing_documents,
    load_existing_pages,
    process_document,
)


def test_process_new_document(tmp_path):
    """Test ingestion of a new text document."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    source_content = "Machine Learning uses data to learn patterns."
    source_file.write_text(source_content, encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    # Act
    result = process_document(
        discovered_file=discovered_file,
        root_directory=root_directory,
        documents_storage_path=documents_path,
        pages_storage_path=pages_path,
    )

    # Assert: ingestion result
    assert result.status == IngestionStatus.NEW_DOCUMENT
    assert result.pages_extracted == 1
    assert result.error is None

    # Assert: document persistence
    documents = load_existing_documents(documents_path)

    assert len(documents) == 1

    document = next(iter(documents.values()))

    assert document.file_name == "sample.txt"
    assert document.content == source_content
    assert document.source_path == "sample.txt"

    # Assert: page persistence
    pages = load_existing_pages(pages_path)

    assert len(pages) == 1
    assert pages[0].document_id == document.document_id
    assert pages[0].page_number == 1
    assert pages[0].content == source_content


def test_process_updated_document(tmp_path):
    """Test updating an existing document and replacing its pages."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    original_content = "Original document content."
    source_file.write_text(original_content, encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    # First ingestion
    first_result = process_document(
        discovered_file,
        root_directory,
        documents_path,
        pages_path,
    )

    # Update source content
    updated_content = "Updated document content with new information."
    source_file.write_text(updated_content, encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    # Act
    result = process_document(
        discovered_file,
        root_directory,
        documents_path,
        pages_path,
    )

    # Assert
    assert first_result.status == IngestionStatus.NEW_DOCUMENT
    assert result.status == IngestionStatus.UPDATED_DOCUMENT
    assert result.document_id == first_result.document_id
    assert result.pages_extracted == 1
    assert result.error is None

    documents = load_existing_documents(documents_path)

    assert len(documents) == 1
    assert documents[result.document_id].content == updated_content

    pages = load_existing_pages(pages_path)

    assert len(pages) == 1
    assert pages[0].document_id == result.document_id
    assert pages[0].content == updated_content


def test_process_unchanged_document(tmp_path):
    """Test skipping an unchanged document without modifying storage."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    source_content = "Machine Learning uses data to learn patterns."
    source_file.write_text(source_content, encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    # First ingestion
    first_result = process_document(
        discovered_file,
        root_directory,
        documents_path,
        pages_path,
    )

    original_documents = documents_path.read_bytes()
    original_pages = pages_path.read_bytes()

    # Act: process the same unchanged document again
    result = process_document(
        discovered_file,
        root_directory,
        documents_path,
        pages_path,
    )

    # Assert
    assert first_result.status == IngestionStatus.NEW_DOCUMENT
    assert result.status == IngestionStatus.SKIPPED_UNCHANGED
    assert result.document_id == first_result.document_id
    assert result.pages_extracted == 1
    assert result.error is None

    assert documents_path.read_bytes() == original_documents
    assert pages_path.read_bytes() == original_pages

    documents = load_existing_documents(documents_path)
    pages = load_existing_pages(pages_path)

    assert len(documents) == 1
    assert len(pages) == 1


def test_process_duplicate_content(tmp_path):
    """Test rejecting identical content from a different source path."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    first_file = root_directory / "first.txt"
    second_file = root_directory / "second.txt"

    shared_content = "Machine Learning uses data to learn patterns."

    first_file.write_text(shared_content, encoding="utf-8")
    second_file.write_text(shared_content, encoding="utf-8")

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    first_discovered = DiscoveredFile(
        path=first_file,
        relative_path="first.txt",
        file_name="first.txt",
        extension=".txt",
        file_size=first_file.stat().st_size,
    )

    second_discovered = DiscoveredFile(
        path=second_file,
        relative_path="second.txt",
        file_name="second.txt",
        extension=".txt",
        file_size=second_file.stat().st_size,
    )

    # Act
    first_result = process_document(
        first_discovered,
        root_directory,
        documents_path,
        pages_path,
    )

    second_result = process_document(
        second_discovered,
        root_directory,
        documents_path,
        pages_path,
    )

    # Assert
    assert first_result.status == IngestionStatus.NEW_DOCUMENT
    assert second_result.status == IngestionStatus.DUPLICATE_CONTENT

    assert first_result.document_id != second_result.document_id

    documents = load_existing_documents(documents_path)
    pages = load_existing_pages(pages_path)

    assert len(documents) == 1
    assert len(pages) == 1

    assert first_result.document_id in documents
    assert second_result.document_id not in documents


def test_process_missing_document(tmp_path):
    """Test handling a source file that no longer exists."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "missing.txt"

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="missing.txt",
        file_name="missing.txt",
        extension=".txt",
        file_size=100,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    # Act
    result = process_document(
        discovered_file,
        root_directory,
        documents_path,
        pages_path,
    )

    # Assert
    assert result.status == IngestionStatus.FAILED
    assert result.document_id
    assert result.source_path == "missing.txt"
    assert result.pages_extracted == 0
    assert result.error is not None

    assert not documents_path.exists()
    assert not pages_path.exists()


def test_process_document_storage_failure(tmp_path):
    """Test that document storage failures are propagated."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    source_file.write_text(
        "Machine Learning document.",
        encoding="utf-8",
    )

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    # Act and Assert
    with patch.object(
        pipeline,
        "persist_documents",
        side_effect=OSError("Simulated storage failure"),
    ):
        with pytest.raises(
            OSError,
            match="Simulated storage failure",
        ):
            process_document(
                discovered_file,
                root_directory,
                documents_path,
                pages_path,
            )

    # Verify that rollback restored the original storage state
    assert not pages_path.exists()
    assert not documents_path.exists()


def test_process_document_restores_existing_pages_on_storage_failure(tmp_path):
    """Restore existing page records when document persistence fails."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    source_file.write_text(
        "Updated Machine Learning document.",
        encoding="utf-8",
    )

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    original_pages_content = (
        b'{"document_id":"old-document","page_number":1,'
        b'"content":"Original content","metadata":{}}\n'
    )

    pages_path.write_bytes(original_pages_content)

    # Act and Assert
    with patch.object(
        pipeline,
        "persist_documents",
        side_effect=OSError("Simulated storage failure"),
    ):
        with pytest.raises(
            OSError,
            match="Simulated storage failure",
        ):
            process_document(
                discovered_file,
                root_directory,
                documents_path,
                pages_path,
            )

    # Verify rollback
    assert pages_path.read_bytes() == original_pages_content
    assert not documents_path.exists()


def test_process_document_raises_runtime_error_when_rollback_fails(tmp_path):
    """Raise a clear error when both persistence and rollback fail."""

    # Arrange
    root_directory = tmp_path / "knowledge_base"
    root_directory.mkdir()

    source_file = root_directory / "sample.txt"
    source_file.write_text(
        "Machine Learning document.",
        encoding="utf-8",
    )

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.txt",
        file_name="sample.txt",
        extension=".txt",
        file_size=source_file.stat().st_size,
    )

    documents_path = tmp_path / "documents.jsonl"
    pages_path = tmp_path / "pages.jsonl"

    # Act and Assert
    with patch.object(
        pipeline,
        "persist_documents",
        side_effect=OSError("Simulated storage failure"),
    ):
        with patch.object(
            pipeline,
            "restore_file_state",
            side_effect=OSError("Simulated rollback failure"),
        ):
            with pytest.raises(
                RuntimeError,
                match="Manual storage recovery is required",
            ):
                process_document(
                    discovered_file,
                    root_directory,
                    documents_path,
                    pages_path,
                )
