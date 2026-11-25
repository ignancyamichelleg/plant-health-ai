"""Model architectures for PlantHealth AI.

Every architecture takes raw 0-255 RGB images and does its own scaling, so the
API only needs preprocess() followed by model.predict().
"""

from tensorflow import keras
from tensorflow.keras import layers

from plant_health.features.augmentation import build_augmentation


def _input_and_augmentation(cfg):
    """Create the image input and, if enabled, the training-only augmentation."""
    image_size = int(cfg["data"]["image_size"])
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    if cfg.get("augmentation", False):
        x = build_augmentation(cfg["seed"])(x)
    return inputs, x


def build_small_cnn(cfg):
    """Small CNN trained from scratch (the baseline)."""
    inputs, x = _input_and_augmentation(cfg)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(float(cfg["training"]["dropout"]), name="classifier_dropout")(x)
    outputs = layers.Dense(
        len(cfg["classes"]), activation="softmax", name="predictions"
    )(x)
    return keras.Model(inputs, outputs, name="planthealth_small_cnn")


def build_mobilenet_v2(cfg, weights="imagenet"):
    """Build MobileNetV2 with a frozen backbone and softmax head."""
    image_size = int(cfg["data"]["image_size"])
    inputs, x = _input_and_augmentation(cfg)

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
    x = layers.Dropout(float(cfg["training"]["dropout"]), name="classifier_dropout")(x)
    outputs = layers.Dense(
        len(cfg["classes"]), activation="softmax", name="predictions"
    )(x)
    return keras.Model(inputs, outputs, name="planthealth_mobilenet_v2")


BACKBONES = {
    "small_cnn": build_small_cnn,
    "mobilenet_v2": build_mobilenet_v2,
}


def build_model(cfg):
    """Build and compile the architecture named by cfg["model"]."""
    model_name = cfg["model"]
    if model_name not in BACKBONES:
        raise ValueError(
            f"Unknown model architecture: {model_name}. "
            f"Available: {sorted(BACKBONES)}"
        )
    model = BACKBONES[model_name](cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model