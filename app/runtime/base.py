"""Runtime interface for the Local AI Assistant."""

from collections.abc import Iterator
from typing import Protocol

from app.schemas import (
    GenerationRequest,
    GenerationResult,
    StreamEvent,
)


class RuntimeInterface(Protocol):
    """Contract implemented by an LLM runtime adapter."""

    def health(self) -> bool:
        """Return whether the runtime service is reachable."""
        ...

    def list_models(self) -> list[str]:
        """Return the model identifiers reported by the runtime."""
        ...

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Execute a non-streaming request and return a normalized result."""
        ...

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[StreamEvent]:
        """Yield ordered streaming events for a generation request."""
        ...
