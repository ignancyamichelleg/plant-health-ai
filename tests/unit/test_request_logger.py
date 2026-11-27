"""Tests for structured prediction request logging."""

import json

from plant_health.serving.request_logger import log_prediction


def write_test_prediction(log_path, request_id="test-001"):
    log_prediction(
        request_id=request_id,
        model_version="test-v1",
        status="ok",
        predicted_class="healthy",
        confidence=0.95,
        latency_ms=45.2,
        brightness=120.0,
        blur=180.0,
        width=640,
        height=480,
        log_path=log_path,
    )


def test_log_prediction_creates_valid_record(tmp_path):
    log_path = tmp_path / "logs" / "requests.jsonl"

    write_test_prediction(log_path)

    records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
    ]

    assert len(records) == 1

    record = records[0]

    assert record["request_id"] == "test-001"
    assert record["model_version"] == "test-v1"
    assert record["status"] == "ok"
    assert record["predicted_class"] == "healthy"
    assert record["confidence"] == 0.95
    assert record["latency_ms"] == 45.2
    assert record["brightness"] == 120.0
    assert record["blur"] == 180.0
    assert record["width"] == 640
    assert record["height"] == 480
    assert "ts" in record

    assert len(record) == 11


def test_log_prediction_appends_records(tmp_path):
    log_path = tmp_path / "requests.jsonl"

    write_test_prediction(log_path, request_id="first")
    write_test_prediction(log_path, request_id="second")

    records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
    ]

    assert len(records) == 2
    assert records[0]["request_id"] == "first"
    assert records[1]["request_id"] == "second"


def test_log_prediction_uses_environment_path(tmp_path, monkeypatch):
    log_path = tmp_path / "custom" / "predictions.jsonl"

    monkeypatch.setenv("PLANT_HEALTH_REQUEST_LOG", str(log_path))

    log_prediction(
        request_id="environment-test",
        model_version="test-v1",
        status="ok",
        predicted_class="healthy",
        confidence=0.9,
        latency_ms=10.0,
        brightness=100.0,
        blur=150.0,
        width=160,
        height=160,
    )

    assert log_path.exists()

    record = json.loads(log_path.read_text().strip())
    assert record["request_id"] == "environment-test"
