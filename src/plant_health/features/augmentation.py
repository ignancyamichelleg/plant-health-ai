"""Training-only augmentation. The layers do nothing at inference time."""
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation(seed: int = 42) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.1, seed=seed),
            layers.RandomZoom(0.1, seed=seed),
            layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        ],
        name="augmentation",
    )