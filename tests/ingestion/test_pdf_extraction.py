"""Tests for PDF extraction."""

import pymupdf
import pytest

from app.ingestion.extractors import (
    ExtractionError,
    extract_document,
    extract_pdf_file,
)


def create_test_pdf(path):
    """Generate a real two-page PDF for extraction tests."""

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text(
            (72, 72),
            "First page extraction test.",
        )

        second_page = document.new_page()
        second_page.insert_text(
            (72, 72),
            "Second page extraction test.",
        )

        document.save(path)


def test_extract_pdf_preserves_page_boundaries(tmp_path):
    source = tmp_path / "sample.pdf"
    create_test_pdf(source)

    result = extract_pdf_file(source)

    assert result.file_type == ".pdf"
    assert result.total_pages == 2
    assert len(result.pages) == 2

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2

    assert "First page extraction test." in result.pages[0].content
    assert "Second page extraction test." in result.pages[1].content


def test_extract_pdf_total_character_count(tmp_path):
    source = tmp_path / "sample.pdf"
    create_test_pdf(source)

    result = extract_pdf_file(source)

    expected = sum(len(page.content) for page in result.pages)

    assert result.total_characters == expected


def test_extract_document_dispatches_pdf(tmp_path):
    source = tmp_path / "sample.pdf"
    create_test_pdf(source)

    result = extract_document(source)

    assert result.file_type == ".pdf"
    assert result.total_pages == 2


def test_invalid_pdf_raises_extraction_error(tmp_path):
    source = tmp_path / "invalid.pdf"
    source.write_bytes(b"This is not a valid PDF.")

    with pytest.raises(ExtractionError):
        extract_pdf_file(source)


def test_password_protected_pdf_is_rejected(tmp_path):
    source = tmp_path / "protected.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Protected content")

        document.save(
            source,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )

    with pytest.raises(ExtractionError, match="Password-protected"):
        extract_pdf_file(source)
