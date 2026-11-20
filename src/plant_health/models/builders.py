"""Model architectures for PlantHealth AI."""

from tensorflow import keras
from tensorflow.keras import layers


def build_mobilenet_v2(cfg, weights="imagenet"):
    """Build MobileNetV2 with a frozen backbone and softmax head."""
    image_size = int(cfg["data"]["image_size"])
    num_classes = len(cfg["classes"])
    dropout = float(cfg["training"]["dropout"])

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="image",
    )

    x = inputs

    if cfg.get("augmentation", False):
        augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ],
            name="augmentation",
        )
        x = augmentation(x)

    # Accept raw RGB pixels in the 0-255 range.
    # MobileNetV2 expects inputs normalized to [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )

    backbone.trainable = False

    # Keep pretrained BatchNormalization layers in inference mode.
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="planthealth_mobilenet_v2",
    )


BACKBONES = {
    "mobilenet_v2": build_mobilenet_v2,
}


def build_model(cfg):
    """Select a model architecture from the experiment configuration."""
    model_name = cfg["model"]

    if model_name not in BACKBONES:
        raise ValueError(
            f"Unknown model architecture: {model_name}. "
            f"Available: {sorted(BACKBONES)}"
        )

    return BACKBONES[model_name](cfg)
