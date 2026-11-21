"""Text preparation utilities for the embedding subsystem."""

from app.embeddings.config import EmbeddingConfig


def prepare_passage_text(
    text: str,
    config: EmbeddingConfig,
) -> str:
    """Prepare chunk text with the configured passage prefix."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not text.strip():
        raise ValueError("text must not be empty or whitespace-only.")

    prefix = config.passage_prefix

    if text.startswith(prefix):
        return text

    return f"{prefix}{text}"
