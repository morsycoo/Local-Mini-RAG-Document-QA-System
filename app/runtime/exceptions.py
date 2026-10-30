"""Runtime exceptions for the Local AI Assistant."""


class RuntimeErrorBase(Exception):
    """Base exception for runtime integration errors."""


class RuntimeConnectionError(RuntimeErrorBase):
    """Raised when the runtime service cannot be reached."""


class RuntimeTimeoutError(RuntimeErrorBase):
    """Raised when a runtime request exceeds its timeout."""


class RuntimeHTTPError(RuntimeErrorBase):
    """Raised when the runtime returns an unsuccessful HTTP status."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class RuntimeResponseError(RuntimeErrorBase):
    """Raised when the runtime returns an invalid response."""


class ModelNotFoundError(RuntimeErrorBase):
    """Raised when a requested model is not available."""


class RuntimeStreamingError(RuntimeErrorBase):
    """Raised when a streaming request cannot be completed correctly."""


class InvalidGenerationRequestError(RuntimeErrorBase):
    """Raised when an invalid generation request reaches the runtime layer."""


class IncompleteGenerationError(RuntimeErrorBase):
    """Raised when generation ends before a complete assistant response is available."""
