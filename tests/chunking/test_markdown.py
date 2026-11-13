"""Tests for Markdown block models."""

import pytest
from pydantic import ValidationError

from app.chunking.markdown import MarkdownBlock


def test_valid_heading_block():
    content = "## Introduction"

    block = MarkdownBlock(
        block_type="heading",
        content=content,
        char_start=0,
        char_end=len(content),
        section_path=["Introduction"],
        metadata={"heading_level": 2},
    )

    assert block.block_type == "heading"
    assert block.content == content
    assert block.char_end - block.char_start == len(content)


def test_valid_paragraph_block():
    content = "This is a paragraph."

    block = MarkdownBlock(
        block_type="paragraph",
        content=content,
        char_start=10,
        char_end=10 + len(content),
    )

    assert block.block_type == "paragraph"
    assert block.char_start == 10


def test_reject_unsupported_block_type():
    with pytest.raises(ValidationError):
        MarkdownBlock(
            block_type="image",
            content="![image](image.png)",
            char_start=0,
            char_end=19,
        )


def test_reject_negative_start_offset():
    with pytest.raises(ValidationError):
        MarkdownBlock(
            block_type="paragraph",
            content="Hello",
            char_start=-1,
            char_end=4,
        )


def test_reject_zero_end_offset():
    with pytest.raises(ValidationError):
        MarkdownBlock(
            block_type="paragraph",
            content="",
            char_start=0,
            char_end=0,
        )


def test_reject_inconsistent_content_length():
    with pytest.raises(ValidationError):
        MarkdownBlock(
            block_type="paragraph",
            content="Hello",
            char_start=0,
            char_end=10,
        )


def test_reject_extra_fields():
    with pytest.raises(ValidationError):
        MarkdownBlock(
            block_type="paragraph",
            content="Hello",
            char_start=0,
            char_end=5,
            unknown_field="unexpected",
        )


def test_mutable_defaults_are_independent():
    first = MarkdownBlock(
        block_type="paragraph",
        content="First",
        char_start=0,
        char_end=5,
    )

    second = MarkdownBlock(
        block_type="paragraph",
        content="Second",
        char_start=0,
        char_end=6,
    )

    first.section_path.append("Introduction")
    first.metadata["source"] = "test"

    assert second.section_path == []
    assert second.metadata == {}


def test_preserve_original_whitespace():
    content = "  Paragraph with surrounding spaces.  "

    block = MarkdownBlock(
        block_type="paragraph",
        content=content,
        char_start=0,
        char_end=len(content),
    )

    assert block.content == content
    assert block.char_end == len(content)
