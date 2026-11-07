"""Pydantic models for text chunking."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkRecord(BaseModel):
    """Represents a chunk of text extracted from a source page."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    chunk_id: str = Field(min_length=1)

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    content: str

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    content_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    section_path: list[str] = Field(default_factory=list)

    strategy: str = Field(min_length=1)
    config_signature: str = Field(min_length=1)
    chunking_version: int = Field(ge=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> "ChunkRecord":
        """Ensure offsets match the chunk content length."""

        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start.")

        expected_length = self.char_end - self.char_start

        if len(self.content) != expected_length:
            raise ValueError("Chunk content length must match its character offsets.")

        return self
