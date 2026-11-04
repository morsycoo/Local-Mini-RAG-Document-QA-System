"""Command-line interface for document ingestion."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from app.ingestion.discovery import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_SUPPORTED_EXTENSIONS,
    discover_documents,
)
from app.ingestion.models import IngestionStatus
from app.ingestion.pipeline import process_document


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Discover, extract, and ingest documents into the knowledge base."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("knowledge base/raw"),
        help="Source directory containing documents.",
    )

    parser.add_argument(
        "--documents-store",
        type=Path,
        default=Path("data/knowledge_base/processed/documents.jsonl"),
        help="Path to the processed documents JSONL file.",
    )

    parser.add_argument(
        "--pages-store",
        type=Path,
        default=Path("data/knowledge_base/processed/pages.jsonl"),
        help="Path to the extracted pages JSONL file.",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_SUPPORTED_EXTENSIONS),
        help="Supported document extensions.",
    )

    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_BYTES / (1024 * 1024),
        help="Maximum accepted file size in megabytes.",
    )

    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Disable recursive directory discovery.",
    )

    return parser


def print_discovery_report(report) -> None:
    """Print document discovery results."""

    print("\nDocument Discovery")
    print("-" * 50)
    print(f"Source directory : {report.root_directory}")
    print(f"Discovered files : {report.discovered_count}")
    print(f"Skipped files    : {report.skipped_count}")

    if report.skipped_files:
        print("\nSkipped Files")

        for issue in report.skipped_files:
            print(f"  - {issue.path}: {issue.reason}")


def print_ingestion_summary(results: list) -> None:
    """Print a summary of document ingestion outcomes."""

    counts = Counter(result.status for result in results)

    print("\nIngestion Summary")
    print("-" * 50)
    print(f"Total processed  : {len(results)}")

    for status in IngestionStatus:
        print(f"{status.value:<20}: {counts[status]}")

    print("\nDocument Results")
    print("-" * 50)

    for result in results:
        print(
            f"[{result.status.value}] "
            f"{result.source_path} "
            f"({result.pages_extracted} pages)"
        )

        if result.error:
            print(f"  Error: {result.error}")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the document ingestion workflow."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.max_file_size_mb <= 0:
        parser.error("--max-file-size-mb must be greater than zero.")

    max_file_size_bytes = int(args.max_file_size_mb * 1024 * 1024)

    try:
        discovery_report = discover_documents(
            root_directory=args.source,
            supported_extensions=frozenset(args.extensions),
            max_file_size_bytes=max_file_size_bytes,
            recursive=not args.non_recursive,
        )

    except (OSError, ValueError) as exc:
        print(f"Discovery failed: {exc}", file=sys.stderr)
        return 1

    print_discovery_report(discovery_report)

    results = []

    for discovered_file in discovery_report.discovered_files:
        try:
            result = process_document(
                discovered_file=discovered_file,
                root_directory=discovery_report.root_directory,
                documents_storage_path=args.documents_store,
                pages_storage_path=args.pages_store,
            )

        except Exception as exc:
            print(
                f"\nIngestion stopped because of a storage or processing error:\n"
                f"{discovered_file.relative_path}: {exc}",
                file=sys.stderr,
            )
            return 2

        results.append(result)

    print_ingestion_summary(results)

    failed_count = sum(result.status == IngestionStatus.FAILED for result in results)

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
