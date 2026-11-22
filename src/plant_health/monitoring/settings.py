"""Load configs/monitoring.yaml, falling back to built-in defaults.

The API imports this too. Inside the Docker image the configs/ folder may be
missing, so the defaults below must match configs/monitoring.yaml. Point
PLANT_HEALTH_MONITORING_CONFIG at another file to override.
"""
import copy
import os
from pathlib import Path

import yaml

from plant_health.config import ROOT, _deep_merge

DEFAULT_CONFIG_PATH = ROOT / "configs" / "monitoring.yaml"

DEFAULTS: dict = {
    "request": {
        "confidence_min": 0.60,
        "brightness_min": 30.0,
        "brightness_max": 230.0,
        "blur_min": 50.0,
    },
    "window": {"size": 50, "min_requests": 20},
    "alerts": {
        "low_quality_rate_max": 0.20,
        "uncertain_rate_max": 0.30,
        "brightness_mean_shift_max": 40.0,
        "blur_median_ratio_min": 0.50,
        "mean_confidence_drop_max": 0.15,
        "class_share_shift_max": 0.25,
        "latency_p95_ms_max": 1000.0,
    },
    "paths": {
        "request_log": "logs/requests.jsonl",
        "reference_stats": "reports/reference_stats.json",
        "alerts_out": "reports/alerts.json",
        "report_html": "reports/monitoring_report.html",
        "report_csv": "reports/monitoring_windows.csv",
    },
}


def load_monitoring_config(path: str | Path | None = None) -> dict:
    """Return the monitoring config: defaults updated with the YAML file, if any."""
    if path is None:
        path = os.getenv("PLANT_HEALTH_MONITORING_CONFIG", str(DEFAULT_CONFIG_PATH))
    path = Path(path)
    config = copy.deepcopy(DEFAULTS)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            config = _deep_merge(config, yaml.safe_load(f) or {})
    return config


def request_status(confidence: float, brightness: float, blur: float, cfg: dict) -> str:
    """Status of one prediction. Quality is checked first: a bad photo is not
    worth a confident answer even if the model happens to be confident."""
    r = cfg["request"]
    if (
        brightness < r["brightness_min"]
        or brightness > r["brightness_max"]
        or blur < r["blur_min"]
    ):
        return "low_quality"
    if confidence < r["confidence_min"]:
        return "uncertain"
    return "ok"
