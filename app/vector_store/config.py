"""Configuration for the ChromaDB vector store."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VectorStoreConfig(BaseModel):
    """Validated configuration for the local ChromaDB vector store."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "knowledge_base" / "vector_store" / "chroma"
    )

    collection_name: str = Field(
        default="hadota_chunks_e5_base_v1",
        min_length=1,
        max_length=63,
    )

    distance_metric: Literal["cosine"] = "cosine"

    embedding_dimension: int = Field(
        default=768,
        gt=0,
    )

    anonymized_telemetry: bool = False

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        """Reject invalid collection names."""

        if value != value.strip():
            raise ValueError("collection_name must not contain surrounding whitespace.")

        if not value[0].isalnum() or not value[-1].isalnum():
            raise ValueError(
                "collection_name must start and end with an alphanumeric character."
            )

        if not all(char.isalnum() or char in "._-" for char in value):
            raise ValueError(
                "collection_name may contain only letters, numbers, dots, "
                "underscores, and hyphens."
            )

        return value
