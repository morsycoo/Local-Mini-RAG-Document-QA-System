"""Models and utilities for Markdown structure parsing."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MarkdownBlockType = Literal[
    "heading",
    "paragraph",
    "unordered_list",
    "ordered_list",
    "table",
    "code",
    "horizontal_rule",
]


class MarkdownBlock(BaseModel):
    """Represents a structurally identified Markdown block."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    block_type: MarkdownBlockType
    content: str

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> "MarkdownBlock":
        """Ensure block offsets match the content length."""

        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start.")

        expected_length = self.char_end - self.char_start

        if len(self.content) != expected_length:
            raise ValueError("Block content length must match its character offsets.")

        return self


# ==========================================
# MARKDOWN PATTERNS
# ==========================================

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)$")

_UNORDERED_LIST_RE = re.compile(r"^ {0,3}[-+*][ \t]+")

_ORDERED_LIST_RE = re.compile(r"^ {0,3}\d{1,9}[.)][ \t]+")

_HORIZONTAL_RULE_RE = re.compile(
    r"^ {0,3}(?:"
    r"\*[ \t]*\*[ \t]*\*[ \t]*"
    r"(?:\*[ \t]*)*"
    r"|-[ \t]*-[ \t]*-[ \t]*(?:-[ \t]*)*"
    r"|_[ \t]*_[ \t]*_[ \t]*(?:_[ \t]*)*"
    r")$"
)

_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


# ==========================================
# SOURCE LINE UTILITIES
# ==========================================


def _build_line_spans(
    content: str,
) -> list[tuple[int, int, str]]:
    """Build line records with original character offsets."""

    lines = []
    offset = 0

    for line in content.splitlines(keepends=True):
        end = offset + len(line)

        lines.append((offset, end, line))

        offset = end

    return lines


def _line_text(line: str) -> str:
    """Remove line endings for classification only."""

    return line.rstrip("\r\n")


def _is_blank(line: str) -> bool:
    return not _line_text(line).strip()


def _heading_match(line: str):
    return _HEADING_RE.match(_line_text(line))


def _is_unordered_list(line: str) -> bool:
    return bool(_UNORDERED_LIST_RE.match(_line_text(line)))


def _is_ordered_list(line: str) -> bool:
    return bool(_ORDERED_LIST_RE.match(_line_text(line)))


def _is_horizontal_rule(line: str) -> bool:
    return bool(_HORIZONTAL_RULE_RE.match(_line_text(line)))


def _is_table_row(line: str) -> bool:
    text = _line_text(line).strip()

    return "|" in text and bool(text)


def _is_table_separator(line: str) -> bool:
    text = _line_text(line).strip()

    if "|" not in text:
        return False

    cells = [cell.strip() for cell in text.strip("|").split("|")]

    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
    )


def _is_table_start(
    lines: list[tuple[int, int, str]],
    index: int,
) -> bool:
    return (
        index + 1 < len(lines)
        and _is_table_row(lines[index][2])
        and _is_table_separator(lines[index + 1][2])
    )


def _is_structural_start(
    lines: list[tuple[int, int, str]],
    index: int,
) -> bool:
    line = lines[index][2]

    return any(
        (
            _heading_match(line),
            _FENCE_OPEN_RE.match(_line_text(line)),
            _is_horizontal_rule(line),
            _is_unordered_list(line),
            _is_ordered_list(line),
            _is_table_start(lines, index),
        )
    )


# ==========================================
# CODE FENCE PARSING
# ==========================================


def _find_fence_end(
    lines: list[tuple[int, int, str]],
    start_index: int,
    fence_character: str,
    minimum_length: int,
) -> tuple[int, bool]:
    """Find a valid closing fence.

    Returns:
        A tuple containing the exclusive ending line index
        and whether a closing fence was found.
    """

    closing_re = re.compile(
        rf"^ {{0,3}}{re.escape(fence_character)}"
        rf"{{{minimum_length},}}[ \t]*$"
    )

    for index in range(start_index + 1, len(lines)):
        if closing_re.fullmatch(_line_text(lines[index][2])):
            return index + 1, True

    return len(lines), False


# ==========================================
# LIST PARSING
# ==========================================


def _is_list_continuation(line: str) -> bool:
    text = _line_text(line)

    return bool(text.strip()) and bool(re.match(r"^(?: {2,}|\t)", text))


def _parse_list_end(
    lines: list[tuple[int, int, str]],
    start_index: int,
    list_type: str,
) -> int:
    """Find the exclusive ending index of a list block."""

    index = start_index

    while index < len(lines):
        line = lines[index][2]

        if _is_blank(line):
            break

        if list_type == "unordered_list":
            if _is_unordered_list(line):
                index += 1
                continue

        if list_type == "ordered_list":
            if _is_ordered_list(line):
                index += 1
                continue

        if _is_list_continuation(line):
            index += 1
            continue

        break

    return index


# ==========================================
# TABLE PARSING
# ==========================================


def _parse_table_end(
    lines: list[tuple[int, int, str]],
    start_index: int,
) -> int:
    """Find the exclusive ending index of a table."""

    index = start_index

    while index < len(lines):
        if not _is_table_row(lines[index][2]):
            break

        index += 1

    return index


# ==========================================
# BLOCK CREATION
# ==========================================


def _create_block(
    source: str,
    lines: list[tuple[int, int, str]],
    start_index: int,
    end_index: int,
    block_type: MarkdownBlockType,
    section_path: list[str],
    metadata: dict[str, Any] | None = None,
) -> MarkdownBlock:
    """Create a block using original source offsets."""

    char_start = lines[start_index][0]
    char_end = lines[end_index - 1][1]

    return MarkdownBlock(
        block_type=block_type,
        content=source[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        section_path=list(section_path),
        metadata=metadata or {},
    )


# ==========================================
# MAIN MARKDOWN PARSER
# ==========================================


def parse_markdown(content: str) -> list[MarkdownBlock]:
    """Parse Markdown text into structurally identified blocks."""

    if not isinstance(content, str):
        raise TypeError("content must be a string.")

    if not content:
        return []

    lines = _build_line_spans(content)

    blocks: list[MarkdownBlock] = []

    heading_stack: list[tuple[int, str]] = []

    index = 0

    while index < len(lines):
        line = lines[index][2]

        if _is_blank(line):
            index += 1
            continue

        section_path = [heading for _, heading in heading_stack]

        # ----------------------------------
        # FENCED CODE BLOCK
        # ----------------------------------

        fence_match = _FENCE_OPEN_RE.match(_line_text(line))

        if fence_match:
            fence = fence_match.group(1)

            fence_character = fence[0]
            fence_length = len(fence)

            info_string = fence_match.group(2).strip()

            end_index, closed = _find_fence_end(
                lines=lines,
                start_index=index,
                fence_character=fence_character,
                minimum_length=fence_length,
            )

            blocks.append(
                _create_block(
                    source=content,
                    lines=lines,
                    start_index=index,
                    end_index=end_index,
                    block_type="code",
                    section_path=section_path,
                    metadata={
                        "fence_character": fence_character,
                        "fence_length": fence_length,
                        "info_string": info_string,
                        "closed": closed,
                    },
                )
            )

            index = end_index
            continue

        # ----------------------------------
        # HEADING
        # ----------------------------------

        heading_match = _heading_match(line)

        if heading_match:
            level = len(heading_match.group(1))

            heading_text = heading_match.group(2).strip()

            # Remove optional closing ATX markers.
            heading_text = re.sub(
                r"[ \t]+#+[ \t]*$",
                "",
                heading_text,
            ).strip()

            heading_stack = [item for item in heading_stack if item[0] < level]

            heading_stack.append((level, heading_text))

            blocks.append(
                _create_block(
                    source=content,
                    lines=lines,
                    start_index=index,
                    end_index=index + 1,
                    block_type="heading",
                    section_path=[heading for _, heading in heading_stack],
                    metadata={
                        "heading_level": level,
                        "heading_text": heading_text,
                    },
                )
            )

            index += 1
            continue

        # ----------------------------------
        # HORIZONTAL RULE
        # ----------------------------------

        if _is_horizontal_rule(line):
            blocks.append(
                _create_block(
                    source=content,
                    lines=lines,
                    start_index=index,
                    end_index=index + 1,
                    block_type="horizontal_rule",
                    section_path=section_path,
                )
            )

            index += 1
            continue

        # ----------------------------------
        # TABLE
        # ----------------------------------

        if _is_table_start(lines, index):
            end_index = _parse_table_end(
                lines,
                index,
            )

            blocks.append(
                _create_block(
                    source=content,
                    lines=lines,
                    start_index=index,
                    end_index=end_index,
                    block_type="table",
                    section_path=section_path,
                    metadata={
                        "row_count": end_index - index,
                    },
                )
            )

            index = end_index
            continue

        # ----------------------------------
        # LISTS
        # ----------------------------------

        if _is_unordered_list(line):
            list_type = "unordered_list"

        elif _is_ordered_list(line):
            list_type = "ordered_list"

        else:
            list_type = None

        if list_type is not None:
            end_index = _parse_list_end(
                lines,
                index,
                list_type,
            )

            blocks.append(
                _create_block(
                    source=content,
                    lines=lines,
                    start_index=index,
                    end_index=end_index,
                    block_type=list_type,
                    section_path=section_path,
                )
            )

            index = end_index
            continue

        # ----------------------------------
        # PARAGRAPHS
        # ----------------------------------

        end_index = index + 1

        while end_index < len(lines):
            next_line = lines[end_index][2]

            if _is_blank(next_line):
                break

            if _is_structural_start(lines, end_index):
                break

            end_index += 1

        blocks.append(
            _create_block(
                source=content,
                lines=lines,
                start_index=index,
                end_index=end_index,
                block_type="paragraph",
                section_path=section_path,
            )
        )

        index = end_index

    return blocks
