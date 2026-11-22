"""End-to-end smoke test of the incident demo, without TensorFlow.

Real images go through the real FastAPI app (quality checks + request log),
then the alert check must fire on the degraded phase.
"""
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from plant_health.monitoring.alerts import run
from plant_health.monitoring.settings import load_monitoring_config
from plant_health.monitoring.simulate import simulate
from plant_health.serving.api import create_app


class FakePredictor:
    model_version = "test-v1"

    def predict(self, image_bytes):
        return {
            "predicted_class": "healthy",
            "confidence": 0.9,
            "probabilities": {"angular_leaf_spot": 0.05, "bean_rust": 0.05, "healthy": 0.9},
            "model_version": self.model_version,
        }


@pytest.fixture
def leaf_images(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        img = Image.effect_noise((300, 300), 50 + 10 * i).convert("RGB")
        img.save(folder / f"leaf_{i}.jpg", quality=95)
    return sorted(folder.glob("*.jpg"))


def test_dark_batch_triggers_alert(tmp_path, monkeypatch, leaf_images):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("PLANT_HEALTH_REQUEST_LOG", str(log_path))

    with TestClient(create_app(predictor=FakePredictor())) as client:

        def post(image_bytes):
            resp = client.post(
                "/predict", files={"file": ("leaf.jpg", image_bytes, "image/jpeg")}
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        responses = simulate(leaf_images, post, normal=25, degraded=25, mode="dark")

    assert all(r["status"] == "ok" for r in responses if r["phase"] == "normal")
    assert all(r["status"] == "low_quality" for r in responses if r["phase"] == "dark")
    assert len(log_path.read_text().splitlines()) == 50

    cfg = load_monitoring_config()
    cfg["paths"].update(
        request_log=str(log_path),
        reference_stats=str(tmp_path / "missing.json"),
        alerts_out=str(tmp_path / "alerts.json"),
    )
    cfg["window"]["size"] = 25
    result = run(cfg)
    assert "low_quality_rate" in {a["rule"] for a in result["alerts"]}
    assert json.loads((tmp_path / "alerts.json").read_text())["alerts"]
