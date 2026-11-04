"""Tests for TXT and Markdown extraction."""

import pytest

from app.ingestion.extractors import (
    ExtractionError,
    UnsupportedDocumentTypeError,
    extract_document,
    extract_markdown_file,
    extract_text_file,
)


def test_extract_text_file(tmp_path):
    source = tmp_path / "sample.txt"
    expected = "Hello world.\nThis is a test."

    source.write_text(expected, encoding="utf-8")

    result = extract_text_file(source)

    assert result.file_type == ".txt"
    assert result.total_pages == 1
    assert result.total_characters == len(expected)
    assert result.pages[0].content == expected


def test_extract_markdown_preserves_structure(tmp_path):
    source = tmp_path / "sample.md"
    expected = "# Heading\n\n- First item\n- Second item"

    source.write_text(expected, encoding="utf-8")

    result = extract_markdown_file(source)

    assert result.pages[0].content == expected
    assert result.metadata["markdown_structure_preserved"] is True


def test_extract_arabic_unicode(tmp_path):
    source = tmp_path / "arabic.txt"
    expected = "الذكاء الاصطناعي ومعالجة اللغة الطبيعية"

    source.write_text(expected, encoding="utf-8")

    result = extract_text_file(source)

    assert result.pages[0].content == expected


def test_extract_utf8_bom(tmp_path):
    source = tmp_path / "bom.txt"
    source.write_text(
        "\ufeffHello world",
        encoding="utf-8",
    )

    result = extract_text_file(source)

    assert result.pages[0].content == "Hello world"


def test_invalid_utf8_raises_extraction_error(tmp_path):
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ExtractionError):
        extract_text_file(source)


def test_unsupported_extension_raises_error(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        extract_document(source)


def test_extract_text_file_rejects_wrong_extension(tmp_path):
    source = tmp_path / "sample.md"
    source.write_text("# Heading", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        extract_text_file(source)


def test_extract_nonexistent_file_raises_error(tmp_path):
    source = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        extract_text_file(source)
