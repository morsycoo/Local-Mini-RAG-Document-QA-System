"""Custom exceptions for the vector store subsystem."""


class VectorStoreError(Exception):
    """Base exception for vector store subsystem failures."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector database cannot be accessed."""


class CollectionConfigurationError(VectorStoreError):
    """Raised when a collection has an incompatible configuration."""


class EmbeddingDimensionMismatchError(VectorStoreError):
    """Raised when embedding dimensions do not match the collection."""


class VectorStoreIntegrityError(VectorStoreError):
    """Raised when stored vector data fails integrity validation."""
