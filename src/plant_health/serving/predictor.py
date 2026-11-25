"""Model loading and inference for the PlantHealth AI serving API."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from plant_health.features.preprocessing import preprocess


class Predictor:
    """Load a trained model once and use it for image classification."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)

        model_path = model_dir / "model.keras"
        metadata_path = model_dir / "metadata.json"

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with metadata_path.open(encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.classes = self.metadata["classes"]
        self.image_size = int(self.metadata["image_size"])
        self.model_version = self.metadata["version"]

        if not isinstance(self.classes, list) or not self.classes:
            raise ValueError("Metadata must contain a non-empty classes list")

        if self.image_size <= 0:
            raise ValueError("Metadata image_size must be positive")

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    def predict(self, image_bytes: bytes) -> dict:
        """Predict a class from encoded image bytes."""

        image = preprocess(
            image_bytes,
            image_size=self.image_size,
        )

        batch = np.expand_dims(image, axis=0)

        predictions = self.model.predict(batch, verbose=0)

        probabilities = np.asarray(predictions[0], dtype=np.float32)

        if probabilities.shape != (len(self.classes),):
            raise ValueError(
                "Model output does not match the configured classes"
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite probabilities")

        if np.any(probabilities < 0) or not np.isclose(
            probabilities.sum(), 1.0, atol=1e-3
        ):
            raise ValueError(
                "Model must return non-negative probabilities summing to 1"
            )

        predicted_index = int(np.argmax(probabilities))

        return {
            "predicted_class": self.classes[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.classes,
                    probabilities,
                    strict=True,
                )
            },
            "model_version": self.model_version,
        }
