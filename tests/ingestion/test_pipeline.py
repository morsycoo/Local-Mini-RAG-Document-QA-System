"""Tests for document ingestion pipeline utilities."""

import hashlib
import json

import pytest

from app.ingestion.discovery import DiscoveredFile
from app.ingestion.extractors import (
    ExtractedPage,
    ExtractionResult,
)
from app.ingestion.models import (
    DocumentRecord,
    IngestionStatus,
    PageRecord,
)
from app.ingestion.pipeline import (
    build_document_records,
    calculate_content_hash,
    calculate_file_hash,
    classify_document,
    generate_document_id,
    load_existing_documents,
    load_existing_pages,
    normalize_text,
    persist_documents,
    persist_jsonl,
    persist_pages,
)

# ---------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------


def create_document(
    document_id: str = "document_001",
    content: str = "Machine Learning content.",
    content_hash: str | None = None,
) -> DocumentRecord:
    """Create a valid document record for testing."""

    calculated_hash = content_hash or calculate_content_hash(content)

    return DocumentRecord(
        document_id=document_id,
        file_name="sample.md",
        source_path="sample.md",
        file_type=".md",
        content=content,
        source_hash=hashlib.sha256(b"source").hexdigest(),
        content_hash=calculated_hash,
        file_size=len(content.encode("utf-8")),
    )


def create_page(
    document_id: str = "document_001",
    page_number: int = 1,
    content: str = "Sample page.",
) -> PageRecord:
    """Create a valid page record for testing."""

    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        content=content,
    )


# ---------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("\ufeffHello", "Hello"),
        ("Line 1\r\nLine 2", "Line 1\nLine 2"),
        ("Line 1\rLine 2", "Line 1\nLine 2"),
        ("Line 1\nLine 2", "Line 1\nLine 2"),
        ("A\r\nB\rC\nD", "A\nB\nC\nD"),
        ("  Hello  ", "  Hello  "),
        ("", ""),
    ],
)
def test_normalize_text(input_text, expected):
    assert normalize_text(input_text) == expected


def test_normalize_text_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(123)


# ---------------------------------------------------------------------
# Source and Content Hashing
# ---------------------------------------------------------------------


def test_calculate_file_hash(tmp_path):
    source_file = tmp_path / "sample.txt"
    source_file.write_bytes(b"Machine Learning")

    expected_hash = hashlib.sha256(b"Machine Learning").hexdigest()

    assert calculate_file_hash(source_file) == expected_hash


def test_calculate_file_hash_changes_with_source_bytes(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_bytes(b"Version 1")
    second_file.write_bytes(b"Version 2")

    assert calculate_file_hash(first_file) != calculate_file_hash(second_file)


def test_calculate_file_hash_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        calculate_file_hash(tmp_path / "missing.txt")


def test_calculate_content_hash_normalizes_newlines():
    first_hash = calculate_content_hash("Line 1\r\nLine 2")
    second_hash = calculate_content_hash("Line 1\nLine 2")

    assert first_hash == second_hash


def test_calculate_content_hash_detects_different_content():
    first_hash = calculate_content_hash("Machine Learning")
    second_hash = calculate_content_hash("Deep Learning")

    assert first_hash != second_hash


def test_calculate_content_hash_returns_sha256():
    result = calculate_content_hash("Test content")

    assert len(result) == 64
    assert all(character in "0123456789abcdef" for character in result)


# ---------------------------------------------------------------------
# Document ID Generation
# ---------------------------------------------------------------------


def test_generate_document_id_is_deterministic(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.md"
    source_file.write_text("Original content", encoding="utf-8")

    first_id = generate_document_id(source_file, root)
    second_id = generate_document_id(source_file, root)

    assert first_id == second_id


def test_generate_document_id_changes_with_relative_path(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    first_file = root / "first.md"
    second_file = root / "second.md"

    first_file.write_text("Same content", encoding="utf-8")
    second_file.write_text("Same content", encoding="utf-8")

    first_id = generate_document_id(first_file, root)
    second_id = generate_document_id(second_file, root)

    assert first_id != second_id


def test_generate_document_id_is_independent_of_file_content(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.md"
    source_file.write_text("Version 1", encoding="utf-8")

    first_id = generate_document_id(source_file, root)

    source_file.write_text("Version 2", encoding="utf-8")

    second_id = generate_document_id(source_file, root)

    assert first_id == second_id


def test_generate_document_id_is_independent_of_root_location(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_root.mkdir()
    second_root.mkdir()

    first_file = first_root / "sample.md"
    second_file = second_root / "sample.md"

    first_file.write_text("Content", encoding="utf-8")
    second_file.write_text("Content", encoding="utf-8")

    assert generate_document_id(first_file, first_root) == generate_document_id(
        second_file, second_root
    )


def test_generate_document_id_rejects_path_outside_root(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    outside_file = tmp_path / "outside.md"
    outside_file.write_text("Content", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the root directory"):
        generate_document_id(outside_file, root)


# ---------------------------------------------------------------------
# Existing Document Loading
# ---------------------------------------------------------------------


def test_load_existing_documents_returns_empty_for_missing_file(tmp_path):
    result = load_existing_documents(tmp_path / "missing.jsonl")

    assert result == {}


def test_load_existing_documents_returns_empty_for_empty_file(tmp_path):
    storage_path = tmp_path / "documents.jsonl"
    storage_path.write_text("", encoding="utf-8")

    assert load_existing_documents(storage_path) == {}


def test_load_existing_documents_restores_valid_records(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    document = create_document()

    storage_path.write_text(
        document.model_dump_json() + "\n",
        encoding="utf-8",
    )

    restored = load_existing_documents(storage_path)

    assert restored == {document.document_id: document}


def test_load_existing_documents_ignores_blank_lines(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    document = create_document()

    storage_path.write_text(
        "\n" + document.model_dump_json() + "\n\n",
        encoding="utf-8",
    )

    restored = load_existing_documents(storage_path)

    assert len(restored) == 1
    assert restored[document.document_id] == document


def test_load_existing_documents_rejects_invalid_json(tmp_path):
    storage_path = tmp_path / "documents.jsonl"
    storage_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_existing_documents(storage_path)


def test_load_existing_documents_rejects_invalid_schema(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    storage_path.write_text(
        '{"document_id": "document_001"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_existing_documents(storage_path)


def test_load_existing_documents_rejects_duplicate_ids(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    document = create_document()

    storage_path.write_text(
        document.model_dump_json() + "\n" + document.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_existing_documents(storage_path)


# ---------------------------------------------------------------------
# Document Classification
# ---------------------------------------------------------------------


def test_classify_new_document():
    document = create_document()

    result = classify_document(document, {})

    assert result == IngestionStatus.NEW_DOCUMENT


def test_classify_unchanged_document():
    existing = create_document()
    incoming = create_document()

    result = classify_document(
        incoming,
        {existing.document_id: existing},
    )

    assert result == IngestionStatus.SKIPPED_UNCHANGED


def test_classify_updated_document():
    existing = create_document(content="Version 1")
    incoming = create_document(content="Version 2")

    result = classify_document(
        incoming,
        {existing.document_id: existing},
    )

    assert result == IngestionStatus.UPDATED_DOCUMENT


def test_classify_duplicate_content():
    existing = create_document(
        document_id="document_001",
        content="Identical content",
    )

    incoming = create_document(
        document_id="document_002",
        content="Identical content",
    )

    result = classify_document(
        incoming,
        {existing.document_id: existing},
    )

    assert result == IngestionStatus.DUPLICATE_CONTENT


def test_classify_changed_source_with_same_content():
    existing = create_document(content="Same content")
    incoming = create_document(content="Same content")

    incoming = incoming.model_copy(
        update={"source_hash": hashlib.sha256(b"changed").hexdigest()}
    )

    result = classify_document(
        incoming,
        {existing.document_id: existing},
    )

    assert result == IngestionStatus.SKIPPED_UNCHANGED


def test_classification_does_not_modify_existing_documents():
    existing = create_document()
    existing_documents = {existing.document_id: existing}

    classify_document(create_document(content="Updated"), existing_documents)

    assert len(existing_documents) == 1
    assert existing_documents[existing.document_id] == existing


# ---------------------------------------------------------------------
# Document Record Construction
# ---------------------------------------------------------------------


def test_build_document_records(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.md"
    source_content = "# Machine Learning\n\nLearning from data."

    source_file.write_text(source_content, encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.md",
        file_name="sample.md",
        extension=".md",
        file_size=source_file.stat().st_size,
    )

    extraction_result = ExtractionResult(
        source_path=str(source_file),
        file_type=".md",
        pages=(
            ExtractedPage(
                page_number=1,
                content=source_content,
            ),
        ),
        total_pages=1,
        total_characters=len(source_content),
    )

    document, pages = build_document_records(
        discovered_file,
        extraction_result,
        root,
    )

    assert isinstance(document, DocumentRecord)
    assert len(pages) == 1
    assert isinstance(pages[0], PageRecord)

    assert document.document_id == generate_document_id(
        source_file,
        root,
    )

    assert document.source_hash == calculate_file_hash(source_file)

    assert document.content_hash == calculate_content_hash(document.content)

    assert document.content == source_content

    assert pages[0].document_id == document.document_id
    assert pages[0].page_number == 1
    assert pages[0].content == source_content


def test_build_document_records_preserves_page_order(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.pdf"
    source_file.write_bytes(b"synthetic pdf source")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.pdf",
        file_name="sample.pdf",
        extension=".pdf",
        file_size=source_file.stat().st_size,
    )

    extraction_result = ExtractionResult(
        source_path=str(source_file),
        file_type=".pdf",
        pages=(
            ExtractedPage(page_number=1, content="First page"),
            ExtractedPage(page_number=2, content="Second page"),
        ),
        total_pages=2,
        total_characters=22,
    )

    document, pages = build_document_records(
        discovered_file,
        extraction_result,
        root,
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert document.content == "First page\n\nSecond page"


def test_build_document_records_rejects_page_count_mismatch(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.md"
    source_file.write_text("Content", encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.md",
        file_name="sample.md",
        extension=".md",
        file_size=source_file.stat().st_size,
    )

    extraction_result = ExtractionResult(
        source_path=str(source_file),
        file_type=".md",
        pages=(ExtractedPage(page_number=1, content="Content"),),
        total_pages=2,
        total_characters=7,
    )

    with pytest.raises(ValueError, match="page count"):
        build_document_records(
            discovered_file,
            extraction_result,
            root,
        )


def test_build_document_records_rejects_file_type_mismatch(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()

    source_file = root / "sample.md"
    source_file.write_text("Content", encoding="utf-8")

    discovered_file = DiscoveredFile(
        path=source_file,
        relative_path="sample.md",
        file_name="sample.md",
        extension=".md",
        file_size=source_file.stat().st_size,
    )

    extraction_result = ExtractionResult(
        source_path=str(source_file),
        file_type=".txt",
        pages=(ExtractedPage(page_number=1, content="Content"),),
        total_pages=1,
        total_characters=7,
    )

    with pytest.raises(ValueError, match="file type"):
        build_document_records(
            discovered_file,
            extraction_result,
            root,
        )


# ---------------------------------------------------------------------
# JSONL Persistence
# ---------------------------------------------------------------------


def test_persist_documents_creates_parent_directory(tmp_path):
    storage_path = tmp_path / "processed" / "documents.jsonl"

    document = create_document()

    count = persist_documents([document], storage_path)

    assert count == 1
    assert storage_path.exists()
    assert storage_path.parent.exists()


def test_persist_jsonl_writes_independent_json_records(tmp_path):
    storage_path = tmp_path / "records.jsonl"

    records = [
        create_page(page_number=1),
        create_page(page_number=2),
    ]

    count = persist_jsonl(records, storage_path)

    lines = storage_path.read_text(encoding="utf-8").splitlines()

    assert count == 2
    assert len(lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_persist_documents_supports_round_trip(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    document = create_document()

    persist_documents([document], storage_path)

    restored = load_existing_documents(storage_path)

    assert restored[document.document_id] == document


def test_persist_pages_supports_round_trip(tmp_path):
    storage_path = tmp_path / "pages.jsonl"

    pages = [
        create_page(page_number=1, content="First page"),
        create_page(page_number=2, content="Second page"),
    ]

    count = persist_pages(pages, storage_path)

    restored = load_existing_pages(storage_path)

    assert count == 2
    assert restored == pages


def test_persist_documents_replaces_previous_snapshot(tmp_path):
    storage_path = tmp_path / "documents.jsonl"

    original = create_document(content="Original content")
    updated = create_document(content="Updated content")

    persist_documents([original], storage_path)
    persist_documents([updated], storage_path)

    restored = load_existing_documents(storage_path)

    assert len(restored) == 1
    assert restored[original.document_id] == updated


def test_persist_empty_collection_creates_empty_file(tmp_path):
    storage_path = tmp_path / "empty.jsonl"

    count = persist_documents([], storage_path)

    assert count == 0
    assert storage_path.exists()
    assert storage_path.stat().st_size == 0


# ---------------------------------------------------------------------
# Existing Page Loading
# ---------------------------------------------------------------------


def test_load_existing_pages_returns_empty_for_missing_file(tmp_path):
    result = load_existing_pages(tmp_path / "missing.jsonl")

    assert result == []


def test_load_existing_pages_returns_empty_for_empty_file(tmp_path):
    storage_path = tmp_path / "pages.jsonl"
    storage_path.write_text("", encoding="utf-8")

    assert load_existing_pages(storage_path) == []


def test_load_existing_pages_restores_valid_records(tmp_path):
    storage_path = tmp_path / "pages.jsonl"

    pages = [
        create_page(page_number=1),
        create_page(page_number=2),
    ]

    persist_pages(pages, storage_path)

    assert load_existing_pages(storage_path) == pages


def test_load_existing_pages_preserves_record_order(tmp_path):
    storage_path = tmp_path / "pages.jsonl"

    pages = [
        create_page(page_number=1),
        create_page(page_number=2),
        create_page(page_number=3),
    ]

    persist_pages(pages, storage_path)

    restored = load_existing_pages(storage_path)

    assert [page.page_number for page in restored] == [1, 2, 3]


def test_load_existing_pages_ignores_blank_lines(tmp_path):
    storage_path = tmp_path / "pages.jsonl"

    first = create_page(page_number=1)
    second = create_page(page_number=2)

    storage_path.write_text(
        first.model_dump_json() + "\n\n" + second.model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert load_existing_pages(storage_path) == [first, second]


def test_load_existing_pages_rejects_invalid_json(tmp_path):
    storage_path = tmp_path / "pages.jsonl"
    storage_path.write_text(
        '{"document_id": invalid}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_existing_pages(storage_path)


def test_load_existing_pages_rejects_invalid_schema(tmp_path):
    storage_path = tmp_path / "pages.jsonl"

    storage_path.write_text(
        '{"document_id": "document_001", "page_number": 0, "content": "Invalid"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_existing_pages(storage_path)
