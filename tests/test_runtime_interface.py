"""Tests for the runtime interface contract."""

from collections.abc import Iterator

from app.runtime.base import RuntimeInterface
from app.schemas import (
    GenerationRequest,
    GenerationResult,
    StreamEvent,
)


def test_runtime_interface_methods_are_defined() -> None:
    """Verify that the protocol exposes all required operations."""

    required_methods = {
        "health",
        "list_models",
        "generate",
        "generate_stream",
    }

    assert required_methods.issubset(RuntimeInterface.__dict__.keys())


def test_runtime_interface_method_annotations() -> None:
    """Verify the public input and output contracts."""

    assert RuntimeInterface.generate.__annotations__["request"] is GenerationRequest
    assert RuntimeInterface.generate.__annotations__["return"] is GenerationResult

    assert (
        RuntimeInterface.generate_stream.__annotations__["return"]
        == Iterator[StreamEvent]
    )
