"""Chunking strategies for document text."""

from app.chunking.models import ChunkRecord


def fixed_window_chunk(
    content: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split text into fixed-size overlapping character windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    if not content:
        return []

    step = chunk_size - overlap

    chunks = []
    start = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        chunks.append((start, end, content[start:end]))

        # Stop when the entire source text is covered.
        if end == len(content):
            break

        start += step

    return chunks
