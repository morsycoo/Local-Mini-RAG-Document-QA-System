"""Pydantic models for document ingestion."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngestionStatus(str, Enum):
    """Supported outcomes of a document ingestion operation."""

    NEW_DOCUMENT = "new_document"
    UPDATED_DOCUMENT = "updated_document"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    DUPLICATE_CONTENT = "duplicate_content"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Represents an ingested source document."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    document_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    file_type: str = Field(min_length=1)

    content: str

    source_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    content_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    file_size: int = Field(ge=0)

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    metadata: dict[str, Any] = Field(default_factory=dict)


class PageRecord(BaseModel):
    """Represents text extracted from an individual document page."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    content: str

    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Represents the outcome of processing a source document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)

    status: IngestionStatus

    pages_extracted: int = Field(default=0, ge=0)

    message: str = ""
    error: str | None = None

    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
