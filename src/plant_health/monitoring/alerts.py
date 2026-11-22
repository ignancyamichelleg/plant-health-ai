"""Alert rules over a window of recent requests.

Labels are not available in production (users do not tell us the true
disease), so every rule uses a proxy signal: input quality, model confidence,
the predicted-class mix, or latency.

    python -m plant_health.monitoring.alerts            # check the last window
    python -m plant_health.monitoring.alerts --fail-on-alert   # exit 1 if any fire
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from plant_health.monitoring.logs import last_window, read_requests
from plant_health.monitoring.reference import load_reference
from plant_health.monitoring.settings import load_monitoring_config

# Response plan: what the team does when each rule fires.
ACTIONS = {
    "low_quality_rate": "Inputs are degraded. Keep rejecting/asking for retakes; "
    "check whether a new device, app change or lighting condition is the cause.",
    "brightness_shift": "Input distribution has shifted. Inspect a sample of "
    "recent requests and the upstream capture path before trusting predictions.",
    "blur_drop": "Photos are much blurrier than training data. Inspect samples; "
    "update capture guidance on the phone page.",
    "uncertain_rate": "Model is unsure on many inputs. Inspect samples; if the "
    "images are valid but new, collect and label them for retraining.",
    "confidence_drop": "Mean confidence fell versus validation. Possible drift: "
    "inspect samples, collect labels, consider retraining or rolling back.",
    "class_share_shift": "Predicted class mix differs from the reference. Check "
    "for a real-world change (season, outbreak) or a model/serving bug.",
    "latency_p95": "Service is slow. Check CPU/memory, model size and concurrent "
    "load; scale up or roll back the last deployment.",
}


@dataclass
class Alert:
    rule: str
    value: float
    threshold: float
    message: str
    action: str


def window_stats(df: pd.DataFrame) -> dict:
    """Summary numbers for one window of requests."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    status = df["status"].value_counts(normalize=True)
    return {
        "n": n,
        "start": df["ts"].iloc[0].isoformat(),
        "end": df["ts"].iloc[-1].isoformat(),
        "low_quality_rate": float(status.get("low_quality", 0.0)),
        "uncertain_rate": float(status.get("uncertain", 0.0)),
        "brightness_mean": float(df["brightness"].mean()),
        "blur_median": float(df["blur"].median()),
        "confidence_mean": float(df["confidence"].mean()),
        "latency_p95_ms": float(df["latency_ms"].quantile(0.95)),
        "class_share": {
            k: float(v)
            for k, v in df["predicted_class"].value_counts(normalize=True).items()
        },
    }


def check_window(df: pd.DataFrame, cfg: dict, reference: dict | None) -> list[Alert]:
    """Return the alerts that fire for this window (empty list = healthy)."""
    if len(df) < cfg["window"]["min_requests"]:
        return []
    a = cfg["alerts"]
    s = window_stats(df)
    alerts: list[Alert] = []

    def fire(rule, value, threshold, message):
        alerts.append(Alert(rule, round(value, 4), threshold, message, ACTIONS[rule]))

    if s["low_quality_rate"] > a["low_quality_rate_max"]:
        fire("low_quality_rate", s["low_quality_rate"], a["low_quality_rate_max"],
             f"{s['low_quality_rate']:.0%} of recent photos were low quality")
    if s["uncertain_rate"] > a["uncertain_rate_max"]:
        fire("uncertain_rate", s["uncertain_rate"], a["uncertain_rate_max"],
             f"{s['uncertain_rate']:.0%} of recent predictions were uncertain")
    if s["latency_p95_ms"] > a["latency_p95_ms_max"]:
        fire("latency_p95", s["latency_p95_ms"], a["latency_p95_ms_max"],
             f"p95 latency is {s['latency_p95_ms']:.0f} ms")

    if reference:
        shift = abs(s["brightness_mean"] - reference["brightness"]["mean"])
        if shift > a["brightness_mean_shift_max"]:
            fire("brightness_shift", shift, a["brightness_mean_shift_max"],
                 f"mean brightness {s['brightness_mean']:.0f} vs reference "
                 f"{reference['brightness']['mean']:.0f}")

        ref_blur = reference["blur"]["median"]
        if ref_blur > 0:
            ratio = s["blur_median"] / ref_blur
            if ratio < a["blur_median_ratio_min"]:
                fire("blur_drop", ratio, a["blur_median_ratio_min"],
                     f"median sharpness is {ratio:.0%} of the reference")

        ref_conf = (reference.get("confidence") or {}).get("mean")
        if ref_conf is not None:
            drop = ref_conf - s["confidence_mean"]
            if drop > a["mean_confidence_drop_max"]:
                fire("confidence_drop", drop, a["mean_confidence_drop_max"],
                     f"mean confidence {s['confidence_mean']:.2f} vs reference "
                     f"{ref_conf:.2f}")

        ref_share = reference.get("class_share") or {}
        classes = set(ref_share) | set(s["class_share"])
        if classes:
            gaps = {
                c: abs(s["class_share"].get(c, 0.0) - ref_share.get(c, 0.0))
                for c in classes
            }
            worst = max(gaps, key=gaps.get)
            if gaps[worst] > a["class_share_shift_max"]:
                fire("class_share_shift", gaps[worst], a["class_share_shift_max"],
                     f"'{worst}' share is {s['class_share'].get(worst, 0.0):.0%} "
                     f"vs reference {ref_share.get(worst, 0.0):.0%}")
    return alerts


def run(cfg: dict) -> dict:
    """Check the latest window from the request log and write alerts.json."""
    paths = cfg["paths"]
    requests = read_requests(paths["request_log"])
    window = last_window(requests, cfg["window"]["size"])
    reference = load_reference(paths["reference_stats"])
    alerts = check_window(window, cfg, reference)
    result = {
        "window": window_stats(window),
        "reference_used": reference is not None,
        "skipped_log_lines": requests.attrs.get("skipped_lines", 0),
        "alerts": [asdict(alert) for alert in alerts],
    }
    out = Path(paths["alerts_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="monitoring YAML")
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    result = run(load_monitoring_config(args.config))
    w = result["window"]
    print(f"Window: {w['n']} requests | reference: {result['reference_used']}")
    if not result["alerts"]:
        print("OK - no alerts")
    for alert in result["alerts"]:
        print(f"ALERT [{alert['rule']}] {alert['message']}\n  -> {alert['action']}")
    if args.fail_on_alert and result["alerts"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
