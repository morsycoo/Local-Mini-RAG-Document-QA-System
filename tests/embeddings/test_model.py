"""Tests for embedding model lifecycle and device management."""

from unittest.mock import Mock

import pytest
import torch

from app.embeddings.config import EmbeddingConfig
from app.embeddings.model import EmbeddingModelManager


@pytest.fixture
def config():
    return EmbeddingConfig(embedding_dimension=3)


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    model = Mock()
    model.device = torch.device("cuda:0")
    model.get_sentence_embedding_dimension.return_value = 3

    constructor = Mock(return_value=model)

    monkeypatch.setattr(
        "app.embeddings.model.SentenceTransformer",
        constructor,
    )

    return constructor, model


def test_auto_selects_cuda_when_available(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "auto"}))

    assert manager._resolve_device() == "cuda"


def test_auto_falls_back_to_cpu_when_cuda_unavailable(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "auto"}))

    assert manager._resolve_device() == "cpu"


def test_explicit_cpu_selection(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cpu"}))

    assert manager._resolve_device() == "cpu"


def test_explicit_cuda_raises_when_unavailable(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cuda"}))

    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        manager._resolve_device()


def test_explicit_cuda_selects_cuda_when_available(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cuda"}))

    assert manager._resolve_device() == "cuda"


def test_selected_device_raises_before_loading(config):
    manager = EmbeddingModelManager(config)

    with pytest.raises(RuntimeError, match="has not been loaded"):
        _ = manager.selected_device


def test_load_passes_configuration_and_device(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    constructor, model = mock_sentence_transformer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "auto"}))

    result = manager.load()

    constructor.assert_called_once_with(
        model_name_or_path=config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
        device="cuda",
    )

    assert result is model
    assert manager.selected_device == "cuda:0"


def test_load_reuses_existing_model(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    constructor, model = mock_sentence_transformer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config)

    first_model = manager.load()
    second_model = manager.load()

    assert first_model is second_model
    assert first_model is model
    constructor.assert_called_once()


def test_get_model_raises_before_loading(config):
    manager = EmbeddingModelManager(config)

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()


def test_get_model_returns_loaded_model(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    _, model = mock_sentence_transformer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config)
    manager.load()

    assert manager.get_model() is model


def test_load_rejects_dimension_mismatch(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    constructor, model = mock_sentence_transformer
    model.get_sentence_embedding_dimension.return_value = 768

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config)

    with pytest.raises(ValueError, match="dimension mismatch"):
        manager.load()

    constructor.assert_called_once()

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()


def test_unload_resets_model_and_selected_device(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    _, model = mock_sentence_transformer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    manager = EmbeddingModelManager(config)
    manager.load()

    assert manager.get_model() is model
    assert manager.selected_device == "cuda:0"

    manager.unload()

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()

    with pytest.raises(RuntimeError, match="has not been loaded"):
        _ = manager.selected_device


def test_load_wraps_sentence_transformer_initialization_error(
    config,
    monkeypatch,
):
    from app.embeddings.exceptions import ModelLoadError

    original_error = OSError("Model files are unavailable.")

    constructor = Mock(side_effect=original_error)

    monkeypatch.setattr(
        "app.embeddings.model.SentenceTransformer",
        constructor,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cpu"}))

    with pytest.raises(ModelLoadError) as exc_info:
        manager.load()

    assert exc_info.value.__cause__ is original_error

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()


def test_load_wraps_embedding_dimension_inspection_error(
    config,
    monkeypatch,
):
    from app.embeddings.exceptions import ModelLoadError

    original_error = RuntimeError("Dimension inspection failed.")

    model = Mock()
    model.get_sentence_embedding_dimension.side_effect = original_error

    constructor = Mock(return_value=model)

    monkeypatch.setattr(
        "app.embeddings.model.SentenceTransformer",
        constructor,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cpu"}))

    with pytest.raises(ModelLoadError) as exc_info:
        manager.load()

    assert exc_info.value.__cause__ is original_error

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()


def test_dimension_mismatch_remains_value_error(
    config,
    monkeypatch,
    mock_sentence_transformer,
):
    _, model = mock_sentence_transformer
    model.get_sentence_embedding_dimension.return_value = 768

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    manager = EmbeddingModelManager(config.model_copy(update={"device": "cpu"}))

    with pytest.raises(ValueError, match="dimension mismatch") as exc_info:
        manager.load()

    from app.embeddings.exceptions import ModelLoadError

    assert not isinstance(exc_info.value, ModelLoadError)

    with pytest.raises(RuntimeError, match="has not been loaded"):
        manager.get_model()
