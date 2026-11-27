"""Unit tests for the MobileNetV2 model architecture."""

import numpy as np
import pytest
from tensorflow import keras

from plant_health.config import ROOT, load_config
from plant_health.models.builders import build_mobilenet_v2


@pytest.fixture(scope="module")
def cfg():
    return load_config(
        str(ROOT / "configs" / "experiments" / "mobilenet_frozen.yaml")
    )


@pytest.fixture(scope="module")
def model(cfg):
    # Avoid downloading ImageNet weights during unit tests.
    return build_mobilenet_v2(cfg, weights=None)


def test_model_input_and_output_shapes(model):
    assert model.input_shape == (None, 160, 160, 3)
    assert model.output_shape == (None, 3)


def test_model_returns_class_probabilities(model):
    images = np.zeros((2, 160, 160, 3), dtype=np.float32)
    predictions = model.predict(images, verbose=0)

    assert predictions.shape == (2, 3)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


def test_backbone_is_frozen(model):
    backbone = next(
        layer
        for layer in model.layers
        if "mobilenetv2" in layer.name.lower()
    )

    assert backbone.trainable is False
    assert len(backbone.trainable_weights) == 0


def test_augmentation_is_present(model):
    layer_names = [layer.name for layer in model.layers]

    assert "augmentation" in layer_names


def test_augmentation_can_be_disabled(cfg):
    config = {
        **cfg,
        "augmentation": False,
    }

    model = build_mobilenet_v2(config, weights=None)
    layer_names = [layer.name for layer in model.layers]

    assert "augmentation" not in layer_names


def test_model_can_be_saved_and_reloaded(model, tmp_path):
    path = tmp_path / "mobilenet.keras"

    model.save(path)

    loaded = keras.models.load_model(path, compile=False)

    assert loaded.input_shape == model.input_shape
    assert loaded.output_shape == model.output_shape

    images = np.zeros((1, 160, 160, 3), dtype=np.float32)

    original_predictions = model.predict(images, verbose=0)
    loaded_predictions = loaded.predict(images, verbose=0)

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-5,
        atol=1e-6,
    )
