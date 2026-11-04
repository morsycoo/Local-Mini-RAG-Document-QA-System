"""Document discovery and source-file validation."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DiscoveredFile:
    """Represents a source file that passed discovery validation."""

    path: Path
    relative_path: str
    file_name: str
    extension: str
    file_size: int


@dataclass(frozen=True)
class DiscoveryIssue:
    """Represents a file excluded during discovery."""

    path: str
    reason: str


@dataclass(frozen=True)
class DiscoveryReport:
    """Summarizes the results of a discovery operation."""

    root_directory: Path
    discovered_files: tuple[DiscoveredFile, ...]
    skipped_files: tuple[DiscoveryIssue, ...]

    @property
    def discovered_count(self) -> int:
        """Return the number of accepted files."""

        return len(self.discovered_files)

    @property
    def skipped_count(self) -> int:
        """Return the number of excluded files."""

        return len(self.skipped_files)


def discover_documents(
    root_directory: str | Path,
    *,
    supported_extensions: frozenset[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    recursive: bool = True,
) -> DiscoveryReport:
    """Discover and validate supported source files."""

    root = Path(root_directory).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    if root.is_symlink():
        raise ValueError(f"Source directory must not be a symbolic link: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {root}")

    if max_file_size_bytes < 1:
        raise ValueError("max_file_size_bytes must be greater than zero.")

    normalized_extensions = frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in supported_extensions
    )

    resolved_root = root.resolve(strict=True)

    candidates = root.rglob("*") if recursive else root.iterdir()

    discovered_files = []
    skipped_files = []

    # Sort candidates to make discovery results reproducible.
    for candidate in sorted(
        candidates,
        key=lambda item: item.relative_to(root).as_posix().casefold(),
    ):
        relative_path = candidate.relative_to(root).as_posix()

        # Ignore directories.
        if candidate.is_dir():
            continue

        # Reject symbolic links.
        if candidate.is_symlink():
            skipped_files.append(DiscoveryIssue(relative_path, "symbolic_link"))
            continue

        # Ignore hidden files.
        if candidate.name.startswith("."):
            skipped_files.append(DiscoveryIssue(relative_path, "hidden_file"))
            continue

        extension = candidate.suffix.lower()

        # Reject unsupported file formats.
        if extension not in normalized_extensions:
            skipped_files.append(DiscoveryIssue(relative_path, "unsupported_extension"))
            continue

        try:
            resolved_path = candidate.resolve(strict=True)

            # Ensure the resolved file remains inside the source directory.
            resolved_path.relative_to(resolved_root)

            if not resolved_path.is_file():
                skipped_files.append(
                    DiscoveryIssue(relative_path, "not_a_regular_file")
                )
                continue

            file_size = resolved_path.stat().st_size

        except (OSError, ValueError):
            skipped_files.append(
                DiscoveryIssue(relative_path, "inaccessible_or_unsafe_path")
            )
            continue

        if file_size == 0:
            skipped_files.append(DiscoveryIssue(relative_path, "empty_file"))
            continue

        if file_size > max_file_size_bytes:
            skipped_files.append(DiscoveryIssue(relative_path, "file_too_large"))
            continue

        discovered_files.append(
            DiscoveredFile(
                path=resolved_path,
                relative_path=relative_path,
                file_name=candidate.name,
                extension=extension,
                file_size=file_size,
            )
        )

    return DiscoveryReport(
        root_directory=resolved_root,
        discovered_files=tuple(discovered_files),
        skipped_files=tuple(skipped_files),
    )
