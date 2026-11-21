"""ChromaDB client and collection management."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from app.embeddings.config import EmbeddingConfig
from app.vector_store.config import VectorStoreConfig
from app.vector_store.exceptions import (
    CollectionConfigurationError,
    VectorStoreConnectionError,
)


class ChromaStore:
    """Manage a persistent ChromaDB client and collection."""

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or VectorStoreConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()

        if self.config.embedding_dimension != self.embedding_config.embedding_dimension:
            raise CollectionConfigurationError(
                "Vector store and embedding model dimensions do not match."
            )

        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        """Return the connected ChromaDB client."""

        if self._client is None:
            raise VectorStoreConnectionError(
                "ChromaDB client is not connected. Call connect() first."
            )

        return self._client

    @property
    def collection(self) -> Any:
        """Return the loaded ChromaDB collection."""

        if self._collection is None:
            raise CollectionConfigurationError(
                "Collection is not loaded. Call get_or_create_collection() first."
            )

        return self._collection

    def connect(self) -> Any:
        """Create or reuse a persistent ChromaDB client."""

        if self._client is not None:
            return self._client

        try:
            self.config.persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=self.config.anonymized_telemetry,
                ),
            )

            return self._client

        except Exception as error:
            raise VectorStoreConnectionError(
                "Failed to initialize the persistent ChromaDB client."
            ) from error

    def _expected_collection_metadata(self) -> dict[str, str | int]:
        """Build metadata identifying the embedding configuration."""

        return {
            "embedding_model": self.embedding_config.model_id,
            "embedding_revision": self.embedding_config.revision,
            "embedding_dimension": self.embedding_config.embedding_dimension,
            "distance_metric": self.config.distance_metric,
        }

    def _validate_collection(self, collection: Any) -> None:
        """Validate an existing collection's configuration."""

        actual_metadata = collection.metadata or {}
        expected_metadata = self._expected_collection_metadata()

        for key, expected_value in expected_metadata.items():
            actual_value = actual_metadata.get(key)

            if actual_value != expected_value:
                raise CollectionConfigurationError(
                    f"Collection metadata mismatch for '{key}': "
                    f"expected {expected_value!r}, got {actual_value!r}."
                )

        configuration = collection.configuration or {}
        hnsw_config = configuration.get("hnsw", {})
        actual_space = hnsw_config.get("space")

        if actual_space != self.config.distance_metric:
            raise CollectionConfigurationError(
                f"Collection distance metric mismatch: "
                f"expected {self.config.distance_metric!r}, "
                f"got {actual_space!r}."
            )

    def get_or_create_collection(self) -> Any:
        """Load a compatible collection or create a new one."""

        client = self.connect()

        try:
            collection = client.get_or_create_collection(
                name=self.config.collection_name,
                configuration={
                    "hnsw": {
                        "space": self.config.distance_metric,
                    }
                },
                metadata=self._expected_collection_metadata(),
                embedding_function=None,
            )

            self._validate_collection(collection)
            self._collection = collection

            return collection

        except CollectionConfigurationError:
            raise

        except Exception as error:
            raise CollectionConfigurationError(
                "Failed to load or create the ChromaDB collection."
            ) from error
