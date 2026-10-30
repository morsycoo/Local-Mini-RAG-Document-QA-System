"""Token counting utilities for local language models."""

from pathlib import Path

from tokenizers import Tokenizer


class TokenCounter:
    """Count tokens using a locally stored tokenizer."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        path = Path(tokenizer_path)

        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file was not found: {path}")

        self._tokenizer = Tokenizer.from_file(str(path))

    def count(self, text: str) -> int:
        """Return the number of tokens in a text string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return len(self._tokenizer.encode(text).ids)

    def __call__(self, text: str) -> int:
        """Allow the counter to be used as a callable."""
        return self.count(text)
