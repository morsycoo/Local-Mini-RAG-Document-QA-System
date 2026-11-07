"""Utilities for tracking lines and detecting Markdown code fences."""

import re
from dataclasses import dataclass

from app.chunking.markdown import MarkdownBlock


@dataclass(frozen=True)
class MarkdownLine:
    """Represents a source line with its original character offsets."""

    content: str
    char_start: int
    char_end: int
    line_number: int


@dataclass(frozen=True)
class CodeFenceRange:
    """Represents a fenced code block in the original source."""

    char_start: int
    char_end: int
    opening_line: int
    closing_line: int | None
    fence_char: str
    fence_length: int
    info_string: str
    closed: bool


def track_markdown_lines(content: str) -> list[MarkdownLine]:
    """Split Markdown text into lines while preserving source offsets."""

    if not content:
        return []

    lines = []
    char_start = 0

    for line_number, line_with_ending in enumerate(
        content.splitlines(keepends=True),
        start=1,
    ):
        char_end = char_start + len(line_with_ending)
        line_content = line_with_ending.rstrip("\r\n")

        lines.append(
            MarkdownLine(
                content=line_content,
                char_start=char_start,
                char_end=char_end,
                line_number=line_number,
            )
        )

        char_start = char_end

    return lines


def detect_code_fences(content: str) -> list[CodeFenceRange]:
    """Detect fenced code blocks while preserving original source offsets."""

    lines = track_markdown_lines(content)

    fences = []
    opening_fence = None

    opening_pattern = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

    for line in lines:
        if opening_fence is None:
            match = opening_pattern.match(line.content)

            if match is None:
                continue

            fence_marker = match.group(1)
            info_string = match.group(2).strip()

            # Backtick fence info strings cannot contain backticks.
            if fence_marker.startswith("`") and "`" in info_string:
                continue

            opening_fence = {
                "char_start": line.char_start,
                "opening_line": line.line_number,
                "fence_char": fence_marker[0],
                "fence_length": len(fence_marker),
                "info_string": info_string,
            }

            continue

        closing_pattern = re.compile(
            rf"^ {{0,3}}"
            rf"{re.escape(opening_fence['fence_char'])}"
            rf"{{{opening_fence['fence_length']},}}"
            rf"[ \t]*$"
        )

        if closing_pattern.match(line.content):
            fences.append(
                CodeFenceRange(
                    char_start=opening_fence["char_start"],
                    char_end=line.char_end,
                    opening_line=opening_fence["opening_line"],
                    closing_line=line.line_number,
                    fence_char=opening_fence["fence_char"],
                    fence_length=opening_fence["fence_length"],
                    info_string=opening_fence["info_string"],
                    closed=True,
                )
            )

            opening_fence = None

    if opening_fence is not None:
        fences.append(
            CodeFenceRange(
                char_start=opening_fence["char_start"],
                char_end=len(content),
                opening_line=opening_fence["opening_line"],
                closing_line=None,
                fence_char=opening_fence["fence_char"],
                fence_length=opening_fence["fence_length"],
                info_string=opening_fence["info_string"],
                closed=False,
            )
        )

    return fences


def detect_headings(content: str) -> list[MarkdownBlock]:
    """Detect ATX headings and preserve their original source offsets."""

    lines = track_markdown_lines(content)
    code_fences = detect_code_fences(content)

    headings = []
    section_stack = []

    fence_index = 0

    heading_pattern = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)\s*#*\s*$")

    for line in lines:
        # Skip lines that belong to fenced code blocks.
        while (
            fence_index < len(code_fences)
            and line.char_start >= code_fences[fence_index].char_end
        ):
            fence_index += 1

        if fence_index < len(code_fences):
            active_fence = code_fences[fence_index]

            if active_fence.char_start <= line.char_start < active_fence.char_end:
                continue

        match = heading_pattern.match(line.content)

        if match is None:
            continue

        heading_level = len(match.group(1))
        heading_text = match.group(2).strip()

        if not heading_text:
            continue

        # Update heading hierarchy.
        section_stack = section_stack[: heading_level - 1]
        section_stack.append(heading_text)

        headings.append(
            MarkdownBlock(
                block_type="heading",
                content=content[line.char_start : line.char_end].rstrip("\r\n"),
                char_start=line.char_start,
                char_end=line.char_start
                + len(content[line.char_start : line.char_end].rstrip("\r\n")),
                section_path=section_stack.copy(),
                metadata={
                    "heading_level": heading_level,
                    "heading_text": heading_text,
                },
            )
        )

    return headings


def detect_paragraphs(content: str) -> list[MarkdownBlock]:
    """Detect paragraph blocks while preserving source offsets."""

    lines = track_markdown_lines(content)
    code_fences = detect_code_fences(content)
    headings = detect_headings(content)

    heading_starts = {heading.char_start for heading in headings}

    unordered_pattern = re.compile(r"^ {0,3}[-+*][ \t]+")
    ordered_pattern = re.compile(r"^ {0,3}\d{1,9}[.)][ \t]+")
    horizontal_rule_pattern = re.compile(
        r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
    )

    paragraphs = []
    paragraph_lines = []
    section_stack = []
    fence_index = 0

    def flush_paragraph():
        if not paragraph_lines:
            return

        start = paragraph_lines[0].char_start
        last_line = paragraph_lines[-1]
        end = last_line.char_start + len(last_line.content)

        paragraphs.append(
            MarkdownBlock(
                block_type="paragraph",
                content=content[start:end],
                char_start=start,
                char_end=end,
                section_path=section_stack.copy(),
            )
        )

        paragraph_lines.clear()

    for line in lines:
        while (
            fence_index < len(code_fences)
            and line.char_start >= code_fences[fence_index].char_end
        ):
            fence_index += 1

        inside_code_fence = (
            fence_index < len(code_fences)
            and code_fences[fence_index].char_start
            <= line.char_start
            < code_fences[fence_index].char_end
        )

        if inside_code_fence:
            flush_paragraph()
            continue

        if not line.content.strip():
            flush_paragraph()
            continue

        if line.char_start in heading_starts:
            flush_paragraph()

            heading = next(
                item for item in headings if item.char_start == line.char_start
            )

            section_stack = heading.section_path.copy()
            continue

        stripped_line = line.content.strip()

        is_unordered_list = unordered_pattern.match(line.content)
        is_ordered_list = ordered_pattern.match(line.content)
        is_horizontal_rule = horizontal_rule_pattern.match(line.content)

        if is_unordered_list or is_ordered_list or is_horizontal_rule:
            flush_paragraph()
            continue

        paragraph_lines.append(line)

    flush_paragraph()

    return paragraphs


def detect_unordered_lists(content: str) -> list[MarkdownBlock]:
    """Detect consecutive unordered list items in Markdown content."""

    lines = track_markdown_lines(content)
    code_fences = detect_code_fences(content)
    headings = detect_headings(content)

    heading_map = {heading.char_start: heading for heading in headings}

    list_pattern = re.compile(r"^ {0,3}[-+*][ \t]+")

    list_blocks = []
    current_items = []
    section_stack = []

    fence_index = 0

    def flush_list():
        if not current_items:
            return

        start = current_items[0].char_start
        last_line = current_items[-1]
        end = last_line.char_start + len(last_line.content)

        list_blocks.append(
            MarkdownBlock(
                block_type="unordered_list",
                content=content[start:end],
                char_start=start,
                char_end=end,
                section_path=section_stack.copy(),
            )
        )

        current_items.clear()

    for line in lines:
        while (
            fence_index < len(code_fences)
            and line.char_start >= code_fences[fence_index].char_end
        ):
            fence_index += 1

        inside_code_fence = (
            fence_index < len(code_fences)
            and code_fences[fence_index].char_start
            <= line.char_start
            < code_fences[fence_index].char_end
        )

        if inside_code_fence:
            flush_list()
            continue

        if line.char_start in heading_map:
            flush_list()
            section_stack = heading_map[line.char_start].section_path.copy()
            continue

        if not line.content.strip():
            flush_list()
            continue

        if list_pattern.match(line.content):
            current_items.append(line)
            continue

        flush_list()

    flush_list()

    return list_blocks


def detect_ordered_lists(content: str) -> list[MarkdownBlock]:
    """Detect consecutive ordered Markdown list items."""

    lines = track_markdown_lines(content)
    code_fences = detect_code_fences(content)
    headings = detect_headings(content)

    heading_map = {heading.char_start: heading for heading in headings}

    list_pattern = re.compile(r"^ {0,3}\d{1,9}[.)][ \t]+")

    list_blocks = []
    current_items = []
    section_stack = []
    fence_index = 0

    def flush_list():
        if not current_items:
            return

        start = current_items[0].char_start
        last_line = current_items[-1]
        end = last_line.char_start + len(last_line.content)

        list_blocks.append(
            MarkdownBlock(
                block_type="ordered_list",
                content=content[start:end],
                char_start=start,
                char_end=end,
                section_path=section_stack.copy(),
            )
        )

        current_items.clear()

    for line in lines:
        while (
            fence_index < len(code_fences)
            and line.char_start >= code_fences[fence_index].char_end
        ):
            fence_index += 1

        inside_code_fence = (
            fence_index < len(code_fences)
            and code_fences[fence_index].char_start
            <= line.char_start
            < code_fences[fence_index].char_end
        )

        if inside_code_fence:
            flush_list()
            continue

        if line.char_start in heading_map:
            flush_list()
            section_stack = heading_map[line.char_start].section_path.copy()
            continue

        if not line.content.strip():
            flush_list()
            continue

        if list_pattern.match(line.content):
            current_items.append(line)
            continue

        flush_list()

    flush_list()

    return list_blocks


def detect_markdown_tables(content: str) -> list[MarkdownBlock]:
    """Detect GitHub Flavored Markdown pipe tables."""

    lines = track_markdown_lines(content)
    code_fences = detect_code_fences(content)
    headings = detect_headings(content)

    heading_map = {heading.char_start: heading for heading in headings}

    def split_cells(line: str) -> list[str]:
        """Split a Markdown table row into individual cells."""
        stripped = line.strip()

        if stripped.startswith("|"):
            stripped = stripped[1:]

        if stripped.endswith("|"):
            stripped = stripped[:-1]

        return [cell.strip() for cell in stripped.split("|")]

    def is_separator_row(line: str) -> bool:
        """Check whether a line is a valid GFM table separator."""
        if "|" not in line:
            return False

        cells = split_cells(line)

        if not cells:
            return False

        return all(re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells)

    def is_table_row(line: str) -> bool:
        """Check whether a line has pipe-table structure."""
        return "|" in line

    tables = []
    section_stack = []
    fence_index = 0
    index = 0

    while index < len(lines):
        line = lines[index]

        while (
            fence_index < len(code_fences)
            and line.char_start >= code_fences[fence_index].char_end
        ):
            fence_index += 1

        inside_code_fence = (
            fence_index < len(code_fences)
            and code_fences[fence_index].char_start
            <= line.char_start
            < code_fences[fence_index].char_end
        )

        if inside_code_fence:
            index += 1
            continue

        if line.char_start in heading_map:
            section_stack = heading_map[line.char_start].section_path.copy()
            index += 1
            continue

        # A table requires a header followed by a valid separator row.
        if (
            index + 1 < len(lines)
            and is_table_row(line.content)
            and is_separator_row(lines[index + 1].content)
        ):
            start = line.char_start
            table_end_index = index + 1

            header_cells = split_cells(line.content)
            separator_cells = split_cells(lines[index + 1].content)

            if len(header_cells) != len(separator_cells):
                index += 1
                continue

            # Include consecutive data rows with matching column counts.
            while table_end_index + 1 < len(lines):
                next_line = lines[table_end_index + 1]

                if not is_table_row(next_line.content):
                    break

                if next_line.char_start in heading_map:
                    break

                next_cells = split_cells(next_line.content)

                if len(next_cells) != len(header_cells):
                    break

                table_end_index += 1

            last_line = lines[table_end_index]
            end = last_line.char_start + len(last_line.content)

            tables.append(
                MarkdownBlock(
                    block_type="table",
                    content=content[start:end],
                    char_start=start,
                    char_end=end,
                    section_path=section_stack.copy(),
                    metadata={
                        "column_count": len(header_cells),
                        "header": header_cells,
                    },
                )
            )

            index = table_end_index + 1
            continue

        index += 1

    return tables
