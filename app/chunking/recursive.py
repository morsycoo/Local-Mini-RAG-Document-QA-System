"""Recursive, structure-aware text chunking."""

import re
from dataclasses import dataclass, field

from app.chunking.markdown import MarkdownBlock, parse_markdown


@dataclass(frozen=True)
class TextSegment:
    """A text segment with source offsets and structural context."""

    content: str
    char_start: int
    char_end: int
    section_path: list[str] = field(default_factory=list)
    block_type: str = "paragraph"


def _validate_configuration(
    chunk_size: int,
    min_chunk_size: int,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if min_chunk_size < 0:
        raise ValueError("min_chunk_size must be non-negative.")

    if min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size cannot exceed chunk_size.")


def _split_by_separator(
    segment: TextSegment,
    separator: str,
) -> list[TextSegment]:
    """Split a segment while preserving original source offsets."""

    if not segment.content:
        return []

    if separator == "":
        return [
            TextSegment(
                content=segment.content[index : index + 1],
                char_start=segment.char_start + index,
                char_end=segment.char_start + index + 1,
                section_path=segment.section_path.copy(),
                block_type=segment.block_type,
            )
            for index in range(len(segment.content))
        ]

    parts = []
    cursor = 0

    for match in re.finditer(separator, segment.content):
        end = match.end()

        if end <= cursor:
            continue

        parts.append(
            TextSegment(
                content=segment.content[cursor:end],
                char_start=segment.char_start + cursor,
                char_end=segment.char_start + end,
                section_path=segment.section_path.copy(),
                block_type=segment.block_type,
            )
        )

        cursor = end

    if cursor < len(segment.content):
        parts.append(
            TextSegment(
                content=segment.content[cursor:],
                char_start=segment.char_start + cursor,
                char_end=segment.char_end,
                section_path=segment.section_path.copy(),
                block_type=segment.block_type,
            )
        )

    return parts


def _recursive_split(
    segment: TextSegment,
    chunk_size: int,
    min_chunk_size: int = 0,
) -> list[TextSegment]:
    """Recursively split oversized text while preserving source offsets."""

    if len(segment.content) <= chunk_size:
        return [segment]

    separators = [
        r"\n\s*\n",
        r"(?<=\n)",
        r"(?<=[.!?])\s+",
    ]

    for separator in separators:
        parts = _split_by_separator(segment, separator)

        if len(parts) <= 1:
            continue

        results = []

        for part in parts:
            if len(part.content) <= chunk_size:
                results.append(part)
            else:
                results.extend(
                    _recursive_split(
                        part,
                        chunk_size,
                        min_chunk_size,
                    )
                )

        if results:
            return results

    # Balanced hard fallback.
    content_length = len(segment.content)
    number_of_chunks = (
        content_length + chunk_size - 1
    ) // chunk_size

    base_size, remainder = divmod(
        content_length,
        number_of_chunks,
    )

    if base_size >= min_chunk_size:
        chunk_lengths = [
            base_size + (1 if index < remainder else 0)
            for index in range(number_of_chunks)
        ]
    else:
        chunk_lengths = [
            min(chunk_size, content_length - start)
            for start in range(0, content_length, chunk_size)
        ]

    results = []
    cursor = 0

    for length in chunk_lengths:
        end = cursor + length

        results.append(
            TextSegment(
                content=segment.content[cursor:end],
                char_start=segment.char_start + cursor,
                char_end=segment.char_start + end,
                section_path=segment.section_path.copy(),
                block_type=segment.block_type,
            )
        )

        cursor = end

    return results
def _merge_segments(
    segments: list[TextSegment],
    chunk_size: int,
) -> list[TextSegment]:
    """Merge adjacent segments while preserving source offsets."""

    if not segments:
        return []

    chunks = []

    current_start = segments[0].char_start
    current_end = segments[0].char_end
    current_content = segments[0].content
    current_section = segments[0].section_path.copy()
    current_type = segments[0].block_type

    for segment in segments[1:]:
        compatible_section = segment.section_path == current_section
        combined_length = segment.char_end - current_start

        if (
            compatible_section
            and combined_length <= chunk_size
            and segment.char_start == current_end
        ):
            current_content += segment.content
            current_end = segment.char_end

            if segment.block_type != current_type:
                current_type = "mixed"

            continue

        chunks.append(
            TextSegment(
                content=current_content,
                char_start=current_start,
                char_end=current_end,
                section_path=current_section,
                block_type=current_type,
            )
        )

        current_start = segment.char_start
        current_end = segment.char_end
        current_content = segment.content
        current_section = segment.section_path.copy()
        current_type = segment.block_type

    chunks.append(
        TextSegment(
            content=current_content,
            char_start=current_start,
            char_end=current_end,
            section_path=current_section,
            block_type=current_type,
        )
    )

    return chunks


def _include_source_gaps(
    segments: list[TextSegment],
    source: str,
    chunk_size: int,
) -> list[TextSegment]:
    """Preserve source characters omitted between parsed Markdown blocks."""

    if not segments:
        if not source:
            return []

        return [
            TextSegment(
                content=source[start : min(start + chunk_size, len(source))],
                char_start=start,
                char_end=min(start + chunk_size, len(source)),
                block_type="whitespace",
            )
            for start in range(0, len(source), chunk_size)
        ]

    result = []
    first = segments[0]

    if first.char_start > 0:
        prefix = source[: first.char_start]

        if len(prefix) + len(first.content) <= chunk_size:
            first = TextSegment(
                content=prefix + first.content,
                char_start=0,
                char_end=first.char_end,
                section_path=first.section_path.copy(),
                block_type=first.block_type,
            )
        else:
            result.extend(
                TextSegment(
                    content=prefix[start : start + chunk_size],
                    char_start=start,
                    char_end=min(start + chunk_size, len(prefix)),
                    block_type="whitespace",
                )
                for start in range(0, len(prefix), chunk_size)
            )

    result.append(first)

    for segment in segments[1:]:
        previous = result[-1]
        gap_start = previous.char_end
        gap_end = segment.char_start

        if gap_start < gap_end:
            gap = source[gap_start:gap_end]

            if len(previous.content) + len(gap) <= chunk_size:
                result[-1] = TextSegment(
                    content=previous.content + gap,
                    char_start=previous.char_start,
                    char_end=gap_end,
                    section_path=previous.section_path.copy(),
                    block_type=previous.block_type,
                )

            elif len(gap) + len(segment.content) <= chunk_size:
                segment = TextSegment(
                    content=gap + segment.content,
                    char_start=gap_start,
                    char_end=segment.char_end,
                    section_path=segment.section_path.copy(),
                    block_type=segment.block_type,
                )

            else:
                result.extend(
                    TextSegment(
                        content=gap[start : start + chunk_size],
                        char_start=gap_start + start,
                        char_end=gap_start
                        + min(
                            start + chunk_size,
                            len(gap),
                        ),
                        block_type="whitespace",
                    )
                    for start in range(0, len(gap), chunk_size)
                )

        result.append(segment)

    last = result[-1]

    if last.char_end < len(source):
        suffix_start = last.char_end
        suffix = source[suffix_start:]

        if len(last.content) + len(suffix) <= chunk_size:
            result[-1] = TextSegment(
                content=last.content + suffix,
                char_start=last.char_start,
                char_end=len(source),
                section_path=last.section_path.copy(),
                block_type=last.block_type,
            )
        else:
            result.extend(
                TextSegment(
                    content=suffix[start : start + chunk_size],
                    char_start=suffix_start + start,
                    char_end=suffix_start
                    + min(
                        start + chunk_size,
                        len(suffix),
                    ),
                    block_type="whitespace",
                )
                for start in range(0, len(suffix), chunk_size)
            )

    return result


def recursive_chunk_text(
    content: str,
    chunk_size: int = 1200,
    min_chunk_size: int = 100,
) -> list[TextSegment]:
    """Recursively split generic text using structural separators."""

    _validate_configuration(chunk_size, min_chunk_size)

    if not content:
        return []

    source = TextSegment(
        content=content,
        char_start=0,
        char_end=len(content),
    )

    segments = _recursive_split(
        source,
        chunk_size,
        min_chunk_size,
    )

    return _merge_segments(segments, chunk_size)


def recursive_chunk_markdown(
    content: str,
    chunk_size: int = 1200,
    min_chunk_size: int = 100,
) -> list[TextSegment]:
    """Recursively split Markdown while preserving structure and coverage."""

    _validate_configuration(chunk_size, min_chunk_size)

    if not content:
        return []

    blocks: list[MarkdownBlock] = parse_markdown(content)

    segments = []

    for block in blocks:
        source = TextSegment(
            content=block.content,
            char_start=block.char_start,
            char_end=block.char_end,
            section_path=block.section_path.copy(),
            block_type=block.block_type,
        )

        if len(block.content) <= chunk_size:
            segments.append(source)
        else:
            segments.extend(
                _recursive_split(
                    source,
                    chunk_size,
                    min_chunk_size,
                )
            )

    segments = _include_source_gaps(
        segments,
        content,
        chunk_size,
    )

    return _merge_segments(segments, chunk_size)
