"""Core text encoding interface for the embedding subsystem."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embeddings.config import EmbeddingConfig
from app.embeddings.exceptions import EmbeddingEncodingError
from app.embeddings.text import prepare_passage_text


class EmbeddingEncoder:
    """Encode individual texts and batches using a Sentence Transformer."""

    def __init__(
        self,
        model: SentenceTransformer,
        config: EmbeddingConfig,
    ) -> None:
        self._model = model
        self._config = config

    def encode_text(self, text: str) -> np.ndarray:
        """Convert one text into a validated float32 embedding vector."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingEncodingError(
                "Failed to generate an embedding for the input text."
            ) from exc

        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (self._config.embedding_dimension,):
            raise ValueError(
                "Unexpected embedding shape: "
                f"expected ({self._config.embedding_dimension},), "
                f"received {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains NaN or infinite values.")

        return vector

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode multiple passage texts into a validated float32 matrix."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings.")

        if not texts:
            raise ValueError("texts must not be empty.")

        prepared_texts = [prepare_passage_text(text, self._config) for text in texts]

        try:
            embeddings = self._model.encode(
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingEncodingError(
                "Failed to generate embeddings for the input texts."
            ) from exc

        matrix = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (
            len(prepared_texts),
            self._config.embedding_dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "Unexpected embedding matrix shape: "
                f"expected {expected_shape}, received {matrix.shape}."
            )

        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains NaN or infinite values.")

        return matrix
