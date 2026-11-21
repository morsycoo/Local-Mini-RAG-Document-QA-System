"""Configuration for the embedding subsystem."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """Validated configuration for the embedding model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    model_id: str = Field(
        default="intfloat/multilingual-e5-base",
        min_length=1,
    )

    revision: str = Field(
        default="ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610",
        pattern=r"^[0-9a-f]{40}$",
    )

    embedding_dimension: int = Field(
        default=768,
        gt=0,
    )

    max_sequence_length: int = Field(
        default=512,
        gt=0,
    )

    query_prefix: str = Field(
        default="query: ",
        min_length=1,
    )

    passage_prefix: str = Field(
        default="passage: ",
        min_length=1,
    )

    normalize_embeddings: bool = True

    local_files_only: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
