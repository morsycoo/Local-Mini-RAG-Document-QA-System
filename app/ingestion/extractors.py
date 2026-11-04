"""Format-specific text extraction utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

SUPPORTED_EXTRACTION_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})


@dataclass(frozen=True)
class ExtractedPage:
    """Represents extracted text from one logical document page."""

    page_number: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Contains the complete output of a document extraction operation."""

    source_path: str
    file_type: str
    pages: tuple[ExtractedPage, ...]
    total_pages: int
    total_characters: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ExtractionError(Exception):
    """Raised when a document cannot be extracted successfully."""


class UnsupportedDocumentTypeError(ExtractionError):
    """Raised when the requested document format is unsupported."""


def _validate_source_path(source_path: str | Path) -> Path:
    """Validate the source path before attempting extraction."""

    path = Path(source_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Source file does not exist: {path}")

    if path.is_symlink():
        raise ExtractionError("Symbolic links are not allowed.")

    if not path.is_file():
        raise ExtractionError(f"Source path is not a regular file: {path}")

    return path


def _build_result(
    path: Path,
    pages: tuple[ExtractedPage, ...],
    metadata: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Build a consistent extraction result from extracted pages."""

    return ExtractionResult(
        source_path=str(path),
        file_type=path.suffix.lower(),
        pages=pages,
        total_pages=len(pages),
        total_characters=sum(len(page.content) for page in pages),
        metadata=metadata or {},
    )


def extract_text_file(source_path: str | Path) -> ExtractionResult:
    """Extract UTF-8 text from a plain text file."""

    path = _validate_source_path(source_path)

    if path.suffix.lower() != ".txt":
        raise UnsupportedDocumentTypeError(
            f"Expected a .txt file, received: {path.suffix}"
        )

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"Unable to decode text file as UTF-8: {path}") from exc
    except OSError as exc:
        raise ExtractionError(f"Unable to read text file: {path}") from exc

    page = ExtractedPage(
        page_number=1,
        content=content,
        metadata={"extraction_method": "utf-8"},
    )

    return _build_result(
        path,
        pages=(page,),
        metadata={"encoding": "utf-8-sig"},
    )


def extract_markdown_file(source_path: str | Path) -> ExtractionResult:
    """Extract Markdown while preserving its original structure."""

    path = _validate_source_path(source_path)

    if path.suffix.lower() != ".md":
        raise UnsupportedDocumentTypeError(
            f"Expected a .md file, received: {path.suffix}"
        )

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(
            f"Unable to decode Markdown file as UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise ExtractionError(f"Unable to read Markdown file: {path}") from exc

    page = ExtractedPage(
        page_number=1,
        content=content,
        metadata={"extraction_method": "utf-8"},
    )

    return _build_result(
        path,
        pages=(page,),
        metadata={
            "encoding": "utf-8-sig",
            "markdown_structure_preserved": True,
        },
    )


def extract_pdf_file(source_path: str | Path) -> ExtractionResult:
    """Extract text from a PDF while preserving page boundaries."""

    path = _validate_source_path(source_path)

    if path.suffix.lower() != ".pdf":
        raise UnsupportedDocumentTypeError(
            f"Expected a .pdf file, received: {path.suffix}"
        )

    extracted_pages: list[ExtractedPage] = []

    try:
        with pymupdf.open(path) as pdf_document:
            if pdf_document.needs_pass:
                raise ExtractionError(
                    f"Password-protected PDF is not supported: {path}"
                )

            for page_index in range(len(pdf_document)):
                pdf_page = pdf_document[page_index]
                page_content: str = str(pdf_page.get_text("text", sort=True))

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        content=page_content,
                        metadata={
                            "extraction_method": "pymupdf",
                        },
                    )
                )

            pdf_metadata = pdf_document.metadata or {}

            metadata = {
                "pdf_page_count": len(pdf_document),
                "pdf_metadata": pdf_metadata,
            }

    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Unable to extract PDF document: {path}") from exc

    return _build_result(
        path,
        pages=tuple(extracted_pages),
        metadata=metadata,
    )


def extract_document(source_path: str | Path) -> ExtractionResult:
    """Select the appropriate extractor based on the file extension."""

    path = Path(source_path)

    extension = path.suffix.lower()

    extractors = {
        ".txt": extract_text_file,
        ".md": extract_markdown_file,
        ".pdf": extract_pdf_file,
    }

    extractor = extractors.get(extension)

    if extractor is None:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document extension: {extension}"
        )

    return extractor(source_path)
