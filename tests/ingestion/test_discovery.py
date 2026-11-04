"""Tests for document discovery and source validation."""

from pathlib import Path

import pytest

from app.ingestion.discovery import discover_documents


def test_discovers_supported_files(tmp_path):
    (tmp_path / "sample.txt").write_text("Text content", encoding="utf-8")
    (tmp_path / "sample.md").write_text("# Markdown", encoding="utf-8")
    (tmp_path / "sample.pdf").write_bytes(b"PDF fixture")

    report = discover_documents(tmp_path)

    assert report.discovered_count == 3
    assert report.skipped_count == 0

    extensions = {item.extension for item in report.discovered_files}

    assert extensions == {".txt", ".md", ".pdf"}


def test_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "sample.docx").write_bytes(b"unsupported")

    report = discover_documents(tmp_path)

    assert report.discovered_count == 0
    assert report.skipped_files[0].reason == "unsupported_extension"


def test_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_bytes(b"")

    report = discover_documents(tmp_path)

    assert report.discovered_count == 0
    assert report.skipped_files[0].reason == "empty_file"


def test_skips_files_exceeding_size_limit(tmp_path):
    (tmp_path / "large.txt").write_text(
        "A" * 100,
        encoding="utf-8",
    )

    report = discover_documents(
        tmp_path,
        max_file_size_bytes=50,
    )

    assert report.discovered_count == 0
    assert report.skipped_files[0].reason == "file_too_large"


def test_discovers_files_recursively(tmp_path):
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    (nested_directory / "sample.md").write_text(
        "Nested content",
        encoding="utf-8",
    )

    report = discover_documents(tmp_path)

    assert report.discovered_count == 1
    assert report.discovered_files[0].relative_path == "nested/sample.md"


def test_non_recursive_discovery(tmp_path):
    (tmp_path / "root.txt").write_text("Root content", encoding="utf-8")

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    (nested_directory / "nested.txt").write_text(
        "Nested content",
        encoding="utf-8",
    )

    report = discover_documents(tmp_path, recursive=False)

    assert report.discovered_count == 1
    assert report.discovered_files[0].file_name == "root.txt"


def test_discovery_order_is_deterministic(tmp_path):
    for filename in ["z.txt", "a.md", "m.pdf"]:
        (tmp_path / filename).write_text("Content", encoding="utf-8")

    report = discover_documents(tmp_path)

    filenames = [item.file_name for item in report.discovered_files]

    assert filenames == ["a.md", "m.pdf", "z.txt"]


def test_missing_directory_raises_error(tmp_path):
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        discover_documents(missing_directory)


def test_file_path_instead_of_directory_raises_error(tmp_path):
    source_file = tmp_path / "sample.txt"
    source_file.write_text("Content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        discover_documents(source_file)


def test_invalid_size_limit_raises_error(tmp_path):
    with pytest.raises(ValueError):
        discover_documents(tmp_path, max_file_size_bytes=0)


def test_custom_extensions_are_supported(tmp_path):
    (tmp_path / "sample.csv").write_text("a,b", encoding="utf-8")

    report = discover_documents(
        tmp_path,
        supported_extensions=frozenset({".csv"}),
    )

    assert report.discovered_count == 1
    assert report.discovered_files[0].extension == ".csv"


def test_hidden_files_are_skipped(tmp_path):
    (tmp_path / ".hidden.md").write_text("Hidden", encoding="utf-8")

    report = discover_documents(tmp_path)

    assert report.discovered_count == 0
    assert report.skipped_files[0].reason == "hidden_file"
