"""Unit tests for monitoring: config, log reading, alerts, reference, report."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from plant_health.monitoring.alerts import check_window, run
from plant_health.monitoring.logs import read_requests
from plant_health.monitoring.quality import image_quality
from plant_health.monitoring.reference import build_reference
from plant_health.monitoring.report import build_report
from plant_health.monitoring.settings import load_monitoring_config, request_status
from plant_health.monitoring.simulate import degrade

REFERENCE = {
    "source": "test",
    "n": 100,
    "brightness": {"mean": 110.0, "median": 110.0},
    "blur": {"median": 400.0},
    "class_share": {"angular_leaf_spot": 0.34, "bean_rust": 0.33, "healthy": 0.33},
    "confidence": {"mean": 0.85},
}
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def make_window(n=50, status="ok", brightness=110.0, blur=400.0,
                confidence=0.85, latency_ms=80.0, classes=CLASSES):
    t0 = datetime(2026, 11, 22, tzinfo=timezone.utc)
    return pd.DataFrame([
        {
            "ts": t0 + timedelta(seconds=i),
            "request_id": str(i),
            "model_version": "v1",
            "status": status,
            "predicted_class": classes[i % len(classes)],
            "confidence": confidence,
            "latency_ms": latency_ms,
            "brightness": brightness,
            "blur": blur,
            "width": 500,
            "height": 500,
        }
        for i in range(n)
    ])


def rules(alerts):
    return {a.rule for a in alerts}


def test_defaults_match_yaml():
    from plant_health.monitoring.settings import DEFAULT_CONFIG_PATH, DEFAULTS
    assert load_monitoring_config(DEFAULT_CONFIG_PATH) == DEFAULTS


def test_missing_config_file_uses_defaults(tmp_path):
    cfg = load_monitoring_config(tmp_path / "nope.yaml")
    assert cfg["request"]["confidence_min"] == 0.60


def test_request_status():
    cfg = load_monitoring_config()
    assert request_status(0.9, 120, 300, cfg) == "ok"
    assert request_status(0.4, 120, 300, cfg) == "uncertain"
    assert request_status(0.9, 10, 300, cfg) == "low_quality"
    assert request_status(0.9, 250, 300, cfg) == "low_quality"
    assert request_status(0.9, 120, 5, cfg) == "low_quality"
    # quality wins over confidence
    assert request_status(0.4, 10, 300, cfg) == "low_quality"


def test_healthy_window_has_no_alerts():
    assert check_window(make_window(), load_monitoring_config(), REFERENCE) == []


def test_small_window_never_alerts():
    window = make_window(n=5, status="low_quality", brightness=10)
    assert check_window(window, load_monitoring_config(), REFERENCE) == []


def test_dark_window_fires_quality_and_brightness():
    window = make_window(status="low_quality", brightness=15.0)
    fired = rules(check_window(window, load_monitoring_config(), REFERENCE))
    assert {"low_quality_rate", "brightness_shift"} <= fired


def test_blurry_window_fires_blur_drop():
    window = make_window(blur=40.0)
    assert "blur_drop" in rules(check_window(window, load_monitoring_config(), REFERENCE))


def test_confidence_and_uncertain_rules():
    window = make_window(status="uncertain", confidence=0.5)
    fired = rules(check_window(window, load_monitoring_config(), REFERENCE))
    assert {"uncertain_rate", "confidence_drop"} <= fired


def test_class_share_shift():
    window = make_window(classes=["healthy"])
    assert "class_share_shift" in rules(
        check_window(window, load_monitoring_config(), REFERENCE)
    )


def test_latency_rule():
    window = make_window(latency_ms=2500.0)
    assert "latency_p95" in rules(check_window(window, load_monitoring_config(), None))


def test_no_reference_skips_drift_rules():
    window = make_window(brightness=15.0, blur=10.0, confidence=0.6)
    assert check_window(window, load_monitoring_config(), None) == []


def test_every_alert_has_an_action():
    window = make_window(status="low_quality", brightness=15.0, blur=10.0,
                         confidence=0.3, latency_ms=3000.0, classes=["healthy"])
    alerts = check_window(window, load_monitoring_config(), REFERENCE)
    assert alerts and all(a.action for a in alerts)


def test_read_requests_skips_bad_lines(tmp_path):
    log = tmp_path / "requests.jsonl"
    good = make_window(n=2).assign(ts=lambda d: d["ts"].map(lambda t: t.isoformat()))
    lines = [json.dumps(r) for r in good.to_dict("records")]
    log.write_text("\n".join([lines[1], "{broken", '{"ts": "x"}', lines[0]]) + "\n")
    df = read_requests(log)
    assert list(df["request_id"]) == ["0", "1"]  # sorted by ts
    assert df.attrs["skipped_lines"] == 2


def test_read_requests_missing_file(tmp_path):
    assert read_requests(tmp_path / "none.jsonl").empty


def test_build_reference_uses_split():
    manifest = pd.DataFrame({
        "split": ["train", "validation", "validation"],
        "label": ["healthy", "healthy", "bean_rust"],
        "readable": [True, True, True],
        "brightness": [10.0, 100.0, 120.0],
        "blur": [1.0, 300.0, 500.0],
    })
    ref = build_reference(manifest)
    assert ref["n"] == 2
    assert ref["brightness"]["mean"] == 110.0
    assert ref["class_share"] == {"bean_rust": 0.5, "healthy": 0.5}
    assert ref["confidence"] is None


def test_degrade_lowers_quality():
    img = Image.effect_noise((256, 256), 60).convert("RGB")
    before = image_quality(img)
    assert image_quality(degrade(img, "dark"))["brightness"] < 30
    assert image_quality(degrade(img, "blur"))["blur"] < before["blur"] * 0.1


def _cfg_in(tmp_path):
    cfg = load_monitoring_config()
    cfg["paths"] = {k: str(tmp_path / Path(v).name) for k, v in cfg["paths"].items()}
    return cfg


def test_run_and_report_write_files(tmp_path):
    cfg = _cfg_in(tmp_path)
    normal = make_window(n=50)
    dark = make_window(n=50, status="low_quality", brightness=15.0)
    dark["ts"] = dark["ts"] + timedelta(hours=1)
    log = pd.concat([normal, dark]).assign(ts=lambda d: d["ts"].map(lambda t: t.isoformat()))
    with open(cfg["paths"]["request_log"], "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in log.to_dict("records"))
    with open(cfg["paths"]["reference_stats"], "w") as f:
        json.dump(REFERENCE, f)

    result = run(cfg)
    assert {"low_quality_rate", "brightness_shift"} <= {a["rule"] for a in result["alerts"]}

    table = build_report(cfg)
    assert list(table["alerts"].astype(bool)) == [False, True]
    assert "monitoring report" in Path(cfg["paths"]["report_html"]).read_text()
