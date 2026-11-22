"""Model builders. Every model takes raw 0-255 images and does its own scaling."""
from collections.abc import Callable

from tensorflow import keras
from tensorflow.keras import layers

from plant_health.features.augmentation import build_augmentation


def small_cnn(cfg: dict, x):
    """Small CNN trained from scratch (the baseline)."""
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dropout(cfg["training"]["dropout"])(x)


# P2 adds "mobilenet_v2" here (a function with the same signature).
BACKBONES: dict[str, Callable] = {"small_cnn": small_cnn}


def build_model(cfg: dict) -> keras.Model:
    size = cfg["data"]["image_size"]
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = inputs
    if cfg["augmentation"]:
        x = build_augmentation(cfg["seed"])(x)
    features = BACKBONES[cfg["model"]](cfg, x)
    outputs = layers.Dense(len(cfg["classes"]), activation="softmax")(features)
    model = keras.Model(inputs, outputs, name=cfg["name"])
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model