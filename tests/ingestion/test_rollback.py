from app.ingestion.pipeline import restore_file_state


def test_restore_existing_file(tmp_path):
    """Restore the previous content of an existing file."""

    storage_path = tmp_path / "pages.jsonl"
    storage_path.write_bytes(b"new content")

    restore_file_state(
        storage_path,
        b"previous content",
    )

    assert storage_path.read_bytes() == b"previous content"


def test_remove_new_file_during_rollback(tmp_path):
    """Remove a file that did not exist before the operation."""

    storage_path = tmp_path / "pages.jsonl"
    storage_path.write_bytes(b"new content")

    restore_file_state(
        storage_path,
        None,
    )

    assert not storage_path.exists()