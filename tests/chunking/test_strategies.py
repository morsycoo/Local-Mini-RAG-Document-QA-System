"""Tests for document chunking strategies."""

import pytest

from app.chunking.strategies import fixed_window_chunk


def test_fixed_window_normal_overlapping_input():
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = fixed_window_chunk(
        content=content,
        chunk_size=10,
        overlap=3,
    )

    assert chunks == [
        (0, 10, "ABCDEFGHIJ"),
        (7, 17, "HIJKLMNOPQ"),
        (14, 24, "OPQRSTUVWX"),
        (21, 26, "VWXYZ"),
    ]


def test_fixed_window_preserves_source_offsets():
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = fixed_window_chunk(
        content=content,
        chunk_size=10,
        overlap=3,
    )

    for start, end, chunk_text in chunks:
        assert 0 <= start < end <= len(content)
        assert end - start == len(chunk_text)


def test_fixed_window_preserves_source_content():
    content = "Machine Learning and Retrieval Augmented Generation."

    chunks = fixed_window_chunk(
        content=content,
        chunk_size=15,
        overlap=5,
    )

    for start, end, chunk_text in chunks:
        assert chunk_text == content[start:end]


def test_fixed_window_short_input():
    chunks = fixed_window_chunk(
        content="HELLO",
        chunk_size=10,
        overlap=2,
    )

    assert chunks == [(0, 5, "HELLO")]


def test_fixed_window_exact_size_input():
    chunks = fixed_window_chunk(
        content="ABCDEFGHIJ",
        chunk_size=10,
        overlap=2,
    )

    assert chunks == [(0, 10, "ABCDEFGHIJ")]


def test_fixed_window_empty_input():
    chunks = fixed_window_chunk(
        content="",
        chunk_size=10,
        overlap=2,
    )

    assert chunks == []


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_fixed_window_rejects_invalid_configuration(
    chunk_size,
    overlap,
):
    with pytest.raises(ValueError):
        fixed_window_chunk(
            content="Test content",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_fixed_window_is_deterministic():
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    first_run = fixed_window_chunk(
        content=content,
        chunk_size=10,
        overlap=3,
    )

    second_run = fixed_window_chunk(
        content=content,
        chunk_size=10,
        overlap=3,
    )

    assert first_run == second_run


def test_fixed_window_covers_entire_source():
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = fixed_window_chunk(
        content=content,
        chunk_size=10,
        overlap=3,
    )

    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(content)

    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous[1]
        current_start = current[0]

        assert current_start <= previous_end
        assert current_start > previous[0]
