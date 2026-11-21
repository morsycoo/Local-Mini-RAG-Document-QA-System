"""Model lifecycle management for the embedding subsystem."""

import torch
from sentence_transformers import SentenceTransformer

from app.embeddings.config import EmbeddingConfig
from app.embeddings.exceptions import ModelLoadError


class EmbeddingModelManager:
    """Load, reuse, and release the embedding model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._selected_device: str | None = None

    def _resolve_device(self) -> str:
        """Resolve the configured device into a usable runtime device."""

        requested_device = self._config.device

        if requested_device == "cpu":
            return "cpu"

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was explicitly requested but is not available."
                )
            return "cuda"

        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        raise ValueError(f"Unsupported device configuration: {requested_device}")

    @property
    def selected_device(self) -> str:
        """Return the actual device used by the loaded model."""

        if self._selected_device is None:
            raise RuntimeError("Embedding model has not been loaded.")

        return self._selected_device

    def load(self) -> SentenceTransformer:
        """Load the configured model once and reuse the loaded instance."""

        if self._model is not None:
            return self._model

        device = self._resolve_device()

        try:
            model = SentenceTransformer(
                model_name_or_path=self._config.model_id,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
                device=device,
            )

            actual_dimension = model.get_sentence_embedding_dimension()

        except Exception as exc:
            raise ModelLoadError(
                "Failed to initialize or inspect the embedding model."
            ) from exc

        if actual_dimension != self._config.embedding_dimension:
            raise ValueError(
                "Model embedding dimension mismatch: "
                f"expected {self._config.embedding_dimension}, "
                f"received {actual_dimension}."
            )

        self._model = model
        self._selected_device = str(model.device)

        return self._model

    def get_model(self) -> SentenceTransformer:
        """Return the loaded model or raise if it has not been initialized."""

        if self._model is None:
            raise RuntimeError("Embedding model has not been loaded.")

        return self._model

    def unload(self) -> None:
        """Release the manager's reference to the loaded model."""

        self._model = None
        self._selected_device = None
