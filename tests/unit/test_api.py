"""HTTP tests for the PlantHealth AI serving API."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from plant_health.serving import api
from plant_health.serving.api import create_app

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


class FakePredictor:
    """Return controlled predictions without loading TensorFlow models."""

    model_version = "test-v1"

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def predict(self, image_bytes):
        assert image_bytes

        remaining = (1.0 - self.confidence) / 2.0

        return {
            "predicted_class": "healthy",
            "confidence": self.confidence,
            "probabilities": {
                "angular_leaf_spot": remaining,
                "bean_rust": remaining,
                "healthy": self.confidence,
            },
            "model_version": self.model_version,
        }


def make_image(format="PNG", color=(80, 150, 50)):
    """Create an in-memory test image."""

    image = Image.new("RGB", (200, 200), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate_request_logs(tmp_path, monkeypatch):
    """Prevent HTTP tests from contaminating operational logs."""
    monkeypatch.setenv(
        "PLANT_HEALTH_REQUEST_LOG",
        str(tmp_path / "requests.jsonl"),
    )


@pytest.fixture
def client():
    app = create_app(predictor=FakePredictor())

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "test-v1",
    }


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "leaf.png",
                make_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] == "healthy"
    assert result["model_version"] == "test-v1"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["status"] in {"ok", "uncertain", "low_quality"}
    assert result["request_id"]
    assert result["latency_ms"] >= 0
    assert 0 <= result["quality"]["brightness"] <= 255
    assert result["quality"]["blur"] >= 0
    assert set(result["probabilities"]) == set(CLASSES)


def test_unsupported_content_type(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_invalid_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "fake.png",
                b"this is not a real PNG",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_oversized_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                b"x" * (8 * 1024 * 1024 + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_missing_file(client):
    response = client.post("/predict")

    assert response.status_code == 422


def test_low_quality_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "dark.png",
                make_image(color=(0, 0, 0)),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "low_quality"


def test_low_quality_takes_precedence_over_uncertainty():
    app = create_app(predictor=FakePredictor(confidence=0.5))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "leaf.png",
                    make_image(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 200
    # A uniform-color image is blurry, so quality takes precedence.
    assert response.json()["status"] == "low_quality"


def test_mismatched_image_format(client):
    """Reject a JPEG image falsely declared as PNG."""
    response = client.post(
        "/predict",
        files={
            "file": (
                "leaf.png",
                make_image(format="JPEG"),
                "image/png",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == (
        "Image content type does not match actual format"
    )


def test_prediction_creates_request_log(client, tmp_path, monkeypatch):
    """Verify that an HTTP prediction writes the monitoring record."""
    import json

    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("PLANT_HEALTH_REQUEST_LOG", str(log_path))

    response = client.post(
        "/predict",
        files={
            "file": (
                "leaf.png",
                make_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert log_path.exists()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    result = response.json()

    assert record["request_id"] == result["request_id"]
    assert record["model_version"] == result["model_version"]
    assert record["predicted_class"] == result["predicted_class"]
    assert record["confidence"] == result["confidence"]
    assert record["status"] == result["status"]
    assert record["width"] == 200
    assert record["height"] == 200
    assert record["brightness"] == result["quality"]["brightness"]
    assert record["blur"] == result["quality"]["blur"]
    assert record["latency_ms"] >= 0
    assert "ts" in record
    assert len(record) == 11



def make_sharp_image():
    """Create a deterministic image with adequate brightness and sharpness."""
    image = Image.new("RGB", (256, 256))
    draw = ImageDraw.Draw(image)

    for y in range(0, 256, 8):
        for x in range(0, 256, 8):
            shade = 50 if (x // 8 + y // 8) % 2 == 0 else 200
            draw.rectangle(
                (x, y, x + 7, y + 7),
                fill=(shade, shade, shade),
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_uncertain_prediction():
    """Low confidence alone must trigger the uncertainty status."""
    app = create_app(predictor=FakePredictor(confidence=0.5))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("leaf.png", make_sharp_image(), "image/png")},
        )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "uncertain"
    assert result["confidence"] == pytest.approx(0.5)
    assert 30 <= result["quality"]["brightness"] <= 230
    assert result["quality"]["blur"] >= 50


def test_high_quality_confident_prediction(client):
    """Adequate quality and confidence must produce an OK status."""
    response = client.post(
        "/predict",
        files={"file": ("leaf.png", make_sharp_image(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_decoded_image_pixel_limit(client, monkeypatch):
    """Reject excessive decoded dimensions even for small uploads."""
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 39_999)

    contents = make_image()

    assert len(contents) < api.MAX_UPLOAD_BYTES

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", contents, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Image exceeds the 25 megapixel limit"
    )


def test_decompression_bomb_error(client, monkeypatch):
    """Convert Pillow's decompression protection into an HTTP 413."""
    contents = make_image()

    def reject_image(*args, **kwargs):
        raise Image.DecompressionBombError("Unsafe image dimensions")

    monkeypatch.setattr(api.Image, "open", reject_image)

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", contents, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Image dimensions exceed safe decoding limits"
    )
