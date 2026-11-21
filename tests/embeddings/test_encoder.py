"""Tests for the embedding encoder."""

from unittest.mock import Mock

import numpy as np
import pytest

from app.embeddings.config import EmbeddingConfig
from app.embeddings.encoder import EmbeddingEncoder


class MockEmbeddingModel:
    """Mock Sentence Transformer model for isolated tests."""

    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))

        if self.output is not None:
            return self.output

        if isinstance(texts, str):
            return np.array(
                [0.1, 0.2, 0.3],
                dtype=np.float64,
            )

        return np.array(
            [[0.1, 0.2, 0.3] for _ in texts],
            dtype=np.float64,
        )


@pytest.fixture
def config():
    return EmbeddingConfig(embedding_dimension=3)


@pytest.fixture
def mock_model():
    return MockEmbeddingModel()


@pytest.fixture
def encoder(mock_model, config):
    return EmbeddingEncoder(mock_model, config)


def test_encodes_multiple_texts(encoder, mock_model):
    texts = ["First document.", "Second document."]

    result = encoder.encode_texts(texts)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    assert len(mock_model.calls) == 1


def test_applies_passage_prefix_to_each_text(encoder, mock_model):
    texts = ["First document.", "Second document."]

    encoder.encode_texts(texts)

    received_texts = mock_model.calls[0][0]

    assert received_texts == [
        "passage: First document.",
        "passage: Second document.",
    ]


def test_does_not_duplicate_existing_prefix(encoder, mock_model):
    texts = ["passage: Existing prefix."]

    encoder.encode_texts(texts)

    assert mock_model.calls[0][0] == texts


def test_preserves_input_order(encoder, mock_model):
    texts = ["First.", "Second.", "Third."]

    encoder.encode_texts(texts)

    assert mock_model.calls[0][0] == [
        "passage: First.",
        "passage: Second.",
        "passage: Third.",
    ]


def test_rejects_empty_collection(encoder, mock_model):
    with pytest.raises(ValueError):
        encoder.encode_texts([])

    assert mock_model.calls == []


@pytest.mark.parametrize("texts", ["single text", b"text", None, 123])
def test_rejects_invalid_collection(encoder, mock_model, texts):
    with pytest.raises(TypeError):
        encoder.encode_texts(texts)

    assert mock_model.calls == []


@pytest.mark.parametrize(
    "texts",
    [
        ["Valid text.", ""],
        ["Valid text.", "   "],
        ["Valid text.", None],
        ["Valid text.", 123],
    ],
)
def test_validates_all_texts_before_encoding(
    encoder,
    mock_model,
    texts,
):
    expected_error = TypeError if not isinstance(texts[1], str) else ValueError

    with pytest.raises(expected_error):
        encoder.encode_texts(texts)

    assert mock_model.calls == []


def test_rejects_incorrect_output_shape(config):
    model = MockEmbeddingModel(output=np.array([[0.1, 0.2]], dtype=np.float32))
    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(ValueError, match="Unexpected embedding matrix shape"):
        encoder.encode_texts(["Example."])


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_rejects_non_finite_embeddings(config, invalid_value):
    model = MockEmbeddingModel(output=np.array([[0.1, invalid_value, 0.3]]))
    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(ValueError, match="NaN or infinite"):
        encoder.encode_texts(["Example."])


def test_respects_normalization_configuration(config):
    model = MockEmbeddingModel()
    encoder = EmbeddingEncoder(model, config)

    encoder.encode_texts(["Example."])

    assert model.calls[0][1]["normalize_embeddings"] is True


def test_encode_text_wraps_model_error(config):
    from app.embeddings.exceptions import EmbeddingEncodingError

    original_error = RuntimeError("GPU inference failed.")

    model = MockEmbeddingModel()
    model.encode = Mock(side_effect=original_error)

    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(EmbeddingEncodingError) as exc_info:
        encoder.encode_text("Example text.")

    assert exc_info.value.__cause__ is original_error


def test_encode_texts_wraps_model_error(config):
    from app.embeddings.exceptions import EmbeddingEncodingError

    original_error = RuntimeError("Batch inference failed.")

    model = MockEmbeddingModel()
    model.encode = Mock(side_effect=original_error)

    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(EmbeddingEncodingError) as exc_info:
        encoder.encode_texts(["First text.", "Second text."])

    assert exc_info.value.__cause__ is original_error


def test_single_text_validation_errors_are_not_wrapped(
    encoder,
    mock_model,
):
    from app.embeddings.exceptions import EmbeddingEncodingError

    with pytest.raises(TypeError):
        encoder.encode_text(123)

    with pytest.raises(ValueError):
        encoder.encode_text("   ")

    assert not mock_model.calls


def test_batch_text_validation_errors_are_not_wrapped(
    encoder,
    mock_model,
):
    from app.embeddings.exceptions import EmbeddingEncodingError

    with pytest.raises(TypeError):
        encoder.encode_texts("Single text.")

    with pytest.raises(ValueError):
        encoder.encode_texts(["Valid text.", "   "])

    assert not mock_model.calls


def test_output_validation_errors_are_not_wrapped(config):
    from app.embeddings.exceptions import EmbeddingEncodingError

    model = MockEmbeddingModel(output=np.array([[0.1, 0.2]], dtype=np.float32))
    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(
        ValueError, match="Unexpected embedding matrix shape"
    ) as exc_info:
        encoder.encode_texts(["Example text."])

    assert not isinstance(exc_info.value, EmbeddingEncodingError)


def test_encodes_single_text(config):
    model = MockEmbeddingModel()
    encoder = EmbeddingEncoder(model, config)

    result = encoder.encode_text("Example text.")

    assert result.shape == (3,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert len(model.calls) == 1
    assert model.calls[0][0] == "Example text."


def test_respects_disabled_normalization():
    config = EmbeddingConfig(
        embedding_dimension=3,
        normalize_embeddings=False,
    )

    model = MockEmbeddingModel()
    encoder = EmbeddingEncoder(model, config)

    encoder.encode_texts(["Example text."])

    assert model.calls[0][1]["normalize_embeddings"] is False


def test_preserves_embedding_output_order(config):
    expected_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    model = MockEmbeddingModel(output=expected_embeddings)
    encoder = EmbeddingEncoder(model, config)

    result = encoder.encode_texts(["First.", "Second.", "Third."])

    np.testing.assert_array_equal(result, expected_embeddings)


def test_single_text_does_not_add_passage_prefix(encoder, mock_model):
    text = "Example document."

    encoder.encode_text(text)

    assert mock_model.calls[0][0] == text


def test_single_encoding_rejects_incorrect_output_shape(config):
    model = MockEmbeddingModel(output=np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(
        ValueError,
        match="Unexpected embedding shape",
    ):
        encoder.encode_text("Example text.")


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_single_encoding_rejects_non_finite_embeddings(
    config,
    invalid_value,
):
    model = MockEmbeddingModel(
        output=np.array(
            [0.1, invalid_value, 0.3],
            dtype=np.float32,
        )
    )
    encoder = EmbeddingEncoder(model, config)

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        encoder.encode_text("Example text.")
