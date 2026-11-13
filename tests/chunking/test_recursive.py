"""Tests for recursive, structure-aware chunking."""

import pytest

from app.chunking.recursive import (
    recursive_chunk_markdown,
    recursive_chunk_text,
)


MARKDOWN_SAMPLE = "\n".join(
    [
        "# Local AI Assistant",
        "",
        "This paragraph explains the purpose of the assistant.",
        "It contains enough text to test recursive splitting.",
        "",
        "## Retrieval Pipeline",
        "",
        "The retrieval pipeline processes documents.",
        "It preserves their original structure and metadata.",
        "",
        "### Chunking Strategy",
        "",
        "Chunking divides long documents into smaller pieces.",
        "Each piece should retain its source offsets.",
        "",
        "```python",
        "def retrieve_documents(query):",
        "    return search_index(query)",
        "```",
        "",
        "- First retrieval step",
        "- Second retrieval step",
        "- Third retrieval step",
        "",
        "| Component | Purpose |",
        "|---|---|",
        "| Parser | Extract structure |",
        "| Chunker | Create chunks |",
        "",
    ]
)


def assert_source_offsets(content, chunks):
    """Verify that every chunk matches its original source slice."""

    for chunk in chunks:
        assert 0 <= chunk.char_start < chunk.char_end <= len(content)

        assert chunk.content == content[chunk.char_start : chunk.char_end]

        assert len(chunk.content) == (chunk.char_end - chunk.char_start)


def assert_complete_source_coverage(content, chunks):
    """Verify that chunks cover every source character."""

    covered_positions = set()

    for chunk in chunks:
        covered_positions.update(range(chunk.char_start, chunk.char_end))

    assert covered_positions == set(range(len(content)))


def test_recursive_text_empty_input():
    assert recursive_chunk_text("") == []


def test_recursive_text_short_input():
    content = "Short text."

    chunks = recursive_chunk_text(
        content,
        chunk_size=100,
        min_chunk_size=20,
    )

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(content)


def test_recursive_text_respects_maximum_chunk_size():
    content = "Recursive chunking preserves source positions. " * 20

    chunks = recursive_chunk_text(
        content,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert len(chunks) > 1

    assert all(len(chunk.content) <= 80 for chunk in chunks)


def test_recursive_text_preserves_source_offsets():
    content = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains additional information."
    )

    chunks = recursive_chunk_text(
        content,
        chunk_size=30,
        min_chunk_size=10,
    )

    assert_source_offsets(content, chunks)


def test_recursive_text_covers_entire_source():
    content = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains additional information."
    )

    chunks = recursive_chunk_text(
        content,
        chunk_size=30,
        min_chunk_size=10,
    )

    assert_complete_source_coverage(content, chunks)


def test_recursive_markdown_preserves_source_offsets():
    chunks = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert chunks
    assert_source_offsets(MARKDOWN_SAMPLE, chunks)


def test_recursive_markdown_covers_entire_source():
    chunks = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert_complete_source_coverage(
        MARKDOWN_SAMPLE,
        chunks,
    )


def test_recursive_markdown_respects_maximum_chunk_size():
    chunks = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert all(len(chunk.content) <= 80 for chunk in chunks)


def test_recursive_markdown_preserves_section_paths():
    chunks = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    expected_path = [
        "Local AI Assistant",
        "Retrieval Pipeline",
        "Chunking Strategy",
    ]

    assert any(chunk.section_path == expected_path for chunk in chunks)


def test_recursive_markdown_preserves_code_block():
    code_block = (
        "```python\n"
        "def retrieve_documents(query):\n"
        "    return search_index(query)\n"
        "```\n"
    )

    chunks = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert any(code_block in chunk.content for chunk in chunks)


def test_recursive_markdown_is_deterministic():
    first_run = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    second_run = recursive_chunk_markdown(
        MARKDOWN_SAMPLE,
        chunk_size=80,
        min_chunk_size=20,
    )

    assert first_run == second_run


@pytest.mark.parametrize(
    "chunk_size, min_chunk_size",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 11),
    ],
)
def test_recursive_chunking_rejects_invalid_configuration(
    chunk_size,
    min_chunk_size,
):
    with pytest.raises(ValueError):
        recursive_chunk_text(
            "Test content",
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
        )

    with pytest.raises(ValueError):
        recursive_chunk_markdown(
            MARKDOWN_SAMPLE,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
        )

def test_recursive_text_hard_fallback_respects_maximum_size():
    content = "A" * 250

    chunks = recursive_chunk_text(
        content,
        chunk_size=80,
        min_chunk_size=30,
    )

    assert len(chunks) == 4
    assert all(30 <= len(chunk.content) <= 80 for chunk in chunks)
    assert_source_offsets(content, chunks)
    assert_complete_source_coverage(content, chunks)


def test_recursive_text_hard_fallback_uses_balanced_chunks():
    content = "A" * 250

    chunks = recursive_chunk_text(
        content,
        chunk_size=100,
        min_chunk_size=50,
    )

    assert [len(chunk.content) for chunk in chunks] == [84, 83, 83]


def test_recursive_text_fallback_when_minimum_is_not_feasible():
    content = "A" * 150

    chunks = recursive_chunk_text(
        content,
        chunk_size=80,
        min_chunk_size=76,
    )

    assert [len(chunk.content) for chunk in chunks] == [80, 70]
    assert_source_offsets(content, chunks)
    assert_complete_source_coverage(content, chunks)


def test_recursive_markdown_empty_input():
    assert recursive_chunk_markdown("") == []


def test_recursive_markdown_whitespace_only_input():
    content = "\n\n   \n\n"

    chunks = recursive_chunk_markdown(
        content,
        chunk_size=10,
        min_chunk_size=2,
    )

    assert_source_offsets(content, chunks)
    assert_complete_source_coverage(content, chunks)


def test_recursive_markdown_large_code_block_preserves_coverage():
    code_block = (
        "```python\n"
        + "\n".join(
            f"result_{index} = process({index})"
            for index in range(30)
        )
        + "\n```"
    )

    chunks = recursive_chunk_markdown(
        code_block,
        chunk_size=100,
        min_chunk_size=30,
    )

    assert len(chunks) > 1
    assert_source_offsets(code_block, chunks)
    assert_complete_source_coverage(code_block, chunks)
    assert all(len(chunk.content) <= 100 for chunk in chunks)


def test_recursive_text_preserves_long_unbroken_content():
    content = "X" * 5000

    chunks = recursive_chunk_text(
        content,
        chunk_size=1200,
        min_chunk_size=100,
    )

    assert len(chunks) == 5
    assert all(len(chunk.content) <= 1200 for chunk in chunks)
    assert_source_offsets(content, chunks)
    assert_complete_source_coverage(content, chunks)

