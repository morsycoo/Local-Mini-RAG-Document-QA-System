"""Tests for document ingestion data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.ingestion.models import (
    DocumentRecord,
    IngestionResult,
    IngestionStatus,
    PageRecord,
)


@pytest.fixture
def valid_document_data():
    """Provide valid document data for model tests."""

    return {
        "document_id": "document_001",
        "file_name": "sample.md",
        "source_path": "knowledge base/raw/sample.md",
        "file_type": ".md",
        "content": "Sample document content.",
        "source_hash": "a" * 64,
        "content_hash": "b" * 64,
        "file_size": 128,
        "metadata": {"source_type": "local_file"},
    }


def test_document_record_creation(valid_document_data):
    document = DocumentRecord(**valid_document_data)

    assert document.document_id == "document_001"
    assert document.file_name == "sample.md"
    assert document.file_size == 128


def test_document_record_generates_utc_timestamp(valid_document_data):
    document = DocumentRecord(**valid_document_data)

    assert document.ingested_at.tzinfo is not None
    assert document.ingested_at.utcoffset() == UTC.utcoffset(None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "invalid"),
        ("content_hash", "invalid"),
        ("document_id", ""),
        ("file_name", ""),
        ("file_size", -1),
    ],
)
def test_document_record_rejects_invalid_values(
    valid_document_data,
    field,
    value,
):
    invalid_data = valid_document_data.copy()
    invalid_data[field] = value

    with pytest.raises(ValidationError):
        DocumentRecord(**invalid_data)


def test_document_record_rejects_unexpected_fields(valid_document_data):
    invalid_data = valid_document_data.copy()
    invalid_data["unexpected_field"] = "not_allowed"

    with pytest.raises(ValidationError):
        DocumentRecord(**invalid_data)


def test_page_record_creation():
    page = PageRecord(
        document_id="document_001",
        page_number=1,
        content="Extracted page content.",
    )

    assert page.page_number == 1
    assert page.document_id == "document_001"


def test_page_record_rejects_zero_page_number():
    with pytest.raises(ValidationError):
        PageRecord(
            document_id="document_001",
            page_number=0,
            content="Invalid page.",
        )


def test_ingestion_result_creation():
    result = IngestionResult(
        document_id="document_001",
        source_path="knowledge base/raw/sample.md",
        status=IngestionStatus.NEW_DOCUMENT,
        pages_extracted=2,
        message="Document ingested successfully.",
    )

    assert result.status == IngestionStatus.NEW_DOCUMENT
    assert result.pages_extracted == 2
    assert result.error is None


def test_ingestion_status_values():
    expected_statuses = {
        "new_document",
        "updated_document",
        "skipped_unchanged",
        "duplicate_content",
        "failed",
    }

    actual_statuses = {status.value for status in IngestionStatus}

    assert actual_statuses == expected_statuses
