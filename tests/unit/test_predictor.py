"""Unit tests for the serving predictor."""

import io
import json

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from plant_health.serving.predictor import Predictor

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


@pytest.fixture
def model_dir(tmp_path):
    """Create a small synthetic model with predictable outputs."""

    inputs = tf.keras.Input(shape=(160, 160, 3))

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(
            [0.0, 1.0, 2.0]
        ),
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.save(tmp_path / "model.keras")

    metadata = {
        "version": "test-v1",
        "classes": CLASSES,
        "image_size": 160,
        "run_id": "synthetic-test",
        "git_sha": "test",
        "data_hash": "test",
        "val_macro_f1": 0.0,
        "val_accuracy": 0.0,
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def image_bytes():
    """Create a valid in-memory PNG image."""

    image = Image.new("RGB", (200, 100), color=(80, 150, 50))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def test_predictor_loads_model(model_dir):
    predictor = Predictor(model_dir)

    assert predictor.classes == CLASSES
    assert predictor.image_size == 160
    assert predictor.model_version == "test-v1"


def test_predictor_returns_probabilities(model_dir, image_bytes):
    predictor = Predictor(model_dir)

    result = predictor.predict(image_bytes)

    assert result["predicted_class"] == "healthy"
    assert result["model_version"] == "test-v1"

    assert set(result["probabilities"]) == set(CLASSES)

    probabilities = list(result["probabilities"].values())

    assert np.isclose(sum(probabilities), 1.0, atol=1e-5)
    assert all(0.0 <= value <= 1.0 for value in probabilities)

    assert np.isclose(
        result["confidence"],
        result["probabilities"]["healthy"],
    )


def test_predictor_missing_model(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({
            "version": "test-v1",
            "classes": CLASSES,
            "image_size": 160,
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Model not found"):
        Predictor(tmp_path)


def test_predictor_missing_metadata(tmp_path):
    (tmp_path / "model.keras").touch()

    with pytest.raises(FileNotFoundError, match="Metadata not found"):
        Predictor(tmp_path)


def test_predictor_rejects_wrong_output_shape(
    model_dir, image_bytes, monkeypatch
):
    predictor = Predictor(model_dir)

    def wrong_prediction(batch, verbose=0):
        return np.array([[0.5, 0.5]], dtype=np.float32)

    monkeypatch.setattr(
        predictor.model,
        "predict",
        wrong_prediction,
    )

    with pytest.raises(ValueError, match="does not match"):
        predictor.predict(image_bytes)


def test_predictor_rejects_invalid_probabilities(
    model_dir, image_bytes, monkeypatch
):
    predictor = Predictor(model_dir)

    def invalid_prediction(batch, verbose=0):
        return np.array([[0.2, 0.3, 0.8]], dtype=np.float32)

    monkeypatch.setattr(
        predictor.model,
        "predict",
        invalid_prediction,
    )

    with pytest.raises(ValueError, match="summing to 1"):
        predictor.predict(image_bytes)
