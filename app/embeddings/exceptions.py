"""Custom exceptions for the embedding subsystem."""


class EmbeddingError(Exception):
    """Base exception for embedding subsystem failures."""


class ModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""


class EmbeddingEncodingError(EmbeddingError):
    """Raised when embedding generation fails."""
