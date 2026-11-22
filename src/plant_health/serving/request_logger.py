"""Structured JSONL logging for prediction requests."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOG_LOCK = Lock()

DEFAULT_LOG_PATH = Path("logs/requests.jsonl")


def log_prediction(
    *,
    request_id: str,
    model_version: str,
    status: str,
    predicted_class: str,
    confidence: float,
    latency_ms: float,
    brightness: float,
    blur: float,
    width: int,
    height: int,
    log_path: Path | None = None,
) -> None:
    """Append prediction metadata without storing image contents."""

    if log_path is None:
        log_path = Path(
            os.getenv("PLANT_HEALTH_REQUEST_LOG", str(DEFAULT_LOG_PATH))
        )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "status": status,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "brightness": brightness,
        "blur": blur,
        "width": width,
        "height": height,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with _LOG_LOCK, log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, allow_nan=False) + "\n")
