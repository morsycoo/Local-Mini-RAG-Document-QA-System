"""Document ingestion pipeline utilities."""

import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from app.ingestion.discovery import DiscoveredFile
from app.ingestion.extractors import (
    ExtractionError,
    ExtractionResult,
    extract_document,
)
from app.ingestion.models import (
    DocumentRecord,
    IngestionResult,
    IngestionStatus,
    PageRecord,
)


def normalize_text(content: str) -> str:
    """Normalize BOM and newline characters without altering other content."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")

    return content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate the SHA-256 hash of the original file bytes."""

    path = Path(file_path)
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_content_hash(content: str) -> str:
    """Calculate the SHA-256 hash of normalized textual content."""

    normalized_content = normalize_text(content)
    content_bytes = normalized_content.encode("utf-8")

    return hashlib.sha256(content_bytes).hexdigest()


def generate_document_id(
    file_path: str | Path,
    root_directory: str | Path,
) -> str:
    """Generate a deterministic document ID from its relative path."""

    path = Path(file_path).resolve()
    root = Path(root_directory).resolve()

    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"File path must be inside the root directory: {path}"
        ) from exc

    normalized_path = relative_path.as_posix()
    path_bytes = normalized_path.encode("utf-8")

    return hashlib.sha256(path_bytes).hexdigest()[:16]


def load_existing_documents(
    storage_path: str | Path,
) -> dict[str, DocumentRecord]:
    """Load and validate existing document records from a JSONL file."""

    path = Path(storage_path)

    if not path.exists():
        return {}

    if not path.is_file():
        raise ValueError(f"Storage path is not a file: {path}")

    documents: dict[str, DocumentRecord] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                document = DocumentRecord.model_validate_json(line)

            except Exception as exc:
                raise ValueError(
                    f"Invalid document record at line {line_number}: {exc}"
                ) from exc

            if document.document_id in documents:
                raise ValueError(
                    f"Duplicate document ID '{document.document_id}' "
                    f"at line {line_number}"
                )

            documents[document.document_id] = document

    return documents


def load_existing_pages(
    storage_path: str | Path,
) -> list[PageRecord]:
    """Load and validate existing page records from a JSONL file."""

    path = Path(storage_path)

    if not path.exists():
        return []

    if not path.is_file():
        raise ValueError(f"Storage path is not a file: {path}")

    pages: list[PageRecord] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                page = PageRecord.model_validate_json(line)

            except Exception as exc:
                raise ValueError(
                    f"Invalid page record at line {line_number}: {exc}"
                ) from exc

            pages.append(page)

    return pages


def classify_document(
    document: DocumentRecord,
    existing_documents: dict[str, DocumentRecord],
) -> IngestionStatus:
    """Determine the ingestion status of a document."""

    existing_document = existing_documents.get(document.document_id)

    if existing_document is not None:
        if existing_document.content_hash == document.content_hash:
            return IngestionStatus.SKIPPED_UNCHANGED

        return IngestionStatus.UPDATED_DOCUMENT

    for existing in existing_documents.values():
        if existing.content_hash == document.content_hash:
            return IngestionStatus.DUPLICATE_CONTENT

    return IngestionStatus.NEW_DOCUMENT


def build_document_records(
    discovered_file: DiscoveredFile,
    extraction_result: ExtractionResult,
    root_directory: str | Path,
) -> tuple[DocumentRecord, tuple[PageRecord, ...]]:
    """Build a document record and its corresponding page records."""

    if len(extraction_result.pages) != extraction_result.total_pages:
        raise ValueError("Extracted page count does not match total_pages.")

    if extraction_result.file_type != discovered_file.extension:
        raise ValueError(
            "Extracted file type does not match discovered file extension."
        )

    document_id = generate_document_id(
        discovered_file.path,
        root_directory,
    )

    source_hash = calculate_file_hash(discovered_file.path)

    normalized_pages = [
        normalize_text(page.content) for page in extraction_result.pages
    ]

    normalized_content = "\n\n".join(normalized_pages)

    content_hash = calculate_content_hash(normalized_content)

    document = DocumentRecord(
        document_id=document_id,
        file_name=discovered_file.file_name,
        source_path=discovered_file.relative_path,
        file_type=discovered_file.extension,
        content=normalized_content,
        source_hash=source_hash,
        content_hash=content_hash,
        file_size=discovered_file.file_size,
        metadata={
            "total_pages": extraction_result.total_pages,
            "total_characters": sum(len(content) for content in normalized_pages),
            "combined_content_characters": len(normalized_content),
            "extraction_metadata": extraction_result.metadata,
        },
    )

    page_records = tuple(
        PageRecord(
            document_id=document_id,
            page_number=page.page_number,
            content=normalized_text,
            metadata=page.metadata,
        )
        for page, normalized_text in zip(
            extraction_result.pages,
            normalized_pages,
        )
    )

    return document, page_records


def persist_jsonl(
    records: Iterable[BaseModel],
    storage_path: str | Path,
) -> int:
    """Persist Pydantic records to a JSONL file atomically."""

    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    record_count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            for record in records:
                temporary_file.write(record.model_dump_json())
                temporary_file.write("\n")
                record_count += 1

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)

        return record_count

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def restore_file_state(
    storage_path: str | Path,
    previous_content: bytes | None,
) -> None:
    """Restore a file to its previous state atomically."""

    path = Path(storage_path)

    # The file did not exist before the operation.
    if previous_content is None:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            temporary_file.write(previous_content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def persist_documents(
    documents: Iterable[DocumentRecord],
    storage_path: str | Path,
) -> int:
    """Persist document records to a JSONL file."""

    return persist_jsonl(documents, storage_path)


def persist_pages(
    pages: Iterable[PageRecord],
    storage_path: str | Path,
) -> int:
    """Persist page records to a JSONL file."""

    return persist_jsonl(pages, storage_path)


def process_document(
    discovered_file: DiscoveredFile,
    root_directory: str | Path,
    documents_storage_path: str | Path,
    pages_storage_path: str | Path,
) -> IngestionResult:
    """Process one discovered document and persist its records."""

    try:
        extraction_result = extract_document(discovered_file.path)

        document, new_pages = build_document_records(
            discovered_file=discovered_file,
            extraction_result=extraction_result,
            root_directory=root_directory,
        )

    except (ExtractionError, OSError, ValueError) as exc:
        return IngestionResult(
            document_id=generate_document_id(
                discovered_file.path,
                root_directory,
            ),
            source_path=discovered_file.relative_path,
            status=IngestionStatus.FAILED,
            pages_extracted=0,
            message="Document processing failed.",
            error=str(exc),
        )

    existing_documents = load_existing_documents(documents_storage_path)

    existing_pages = load_existing_pages(pages_storage_path)

    status = classify_document(
        document=document,
        existing_documents=existing_documents,
    )

    if status == IngestionStatus.SKIPPED_UNCHANGED:
        return IngestionResult(
            document_id=document.document_id,
            source_path=document.source_path,
            status=status,
            pages_extracted=len(new_pages),
            message="Document content is unchanged. No updates were made.",
        )

    if status == IngestionStatus.DUPLICATE_CONTENT:
        return IngestionResult(
            document_id=document.document_id,
            source_path=document.source_path,
            status=status,
            pages_extracted=len(new_pages),
            message="Identical content already exists under another document ID.",
        )

    updated_documents = dict(existing_documents)
    updated_documents[document.document_id] = document

    updated_pages = [
        page for page in existing_pages if page.document_id != document.document_id
    ]

    updated_pages.extend(new_pages)

    pages_path = Path(pages_storage_path)

    previous_pages_content = (
        pages_path.read_bytes()
        if pages_path.exists()
        else None
    )

    persist_pages(
        updated_pages,
        pages_storage_path,
    )

    try:
        persist_documents(
            updated_documents.values(),
            documents_storage_path,
        )

    except Exception:
        try:
            restore_file_state(
                pages_storage_path,
                previous_pages_content,
            )
        except Exception as rollback_error:
            raise RuntimeError(
                "Document persistence failed and page rollback "
                "also failed. Manual storage recovery is required."
            ) from rollback_error

        raise

    message = (
        "Document successfully ingested."
        if status == IngestionStatus.NEW_DOCUMENT
        else "Document successfully updated."
    )

    return IngestionResult(
        document_id=document.document_id,
        source_path=document.source_path,
        status=status,
        pages_extracted=len(new_pages),
        message=message,
    )
