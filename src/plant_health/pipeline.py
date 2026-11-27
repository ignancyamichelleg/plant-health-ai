"""End-to-end training pipeline.

Stages: (download) -> manifest -> validate -> train -> reference stats.
A failed validation stops the pipeline before any training happens.

Usage:
    python pipelines/train_pipeline.py --config configs/experiments/baseline_cnn.yaml
"""
import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from plant_health.config import load_config
from plant_health.data.manifest import build_manifest
from plant_health.data.validate import validate, write_report
from plant_health.models.train import run as train_run
from plant_health.monitoring.reference import build_reference_stats

log = logging.getLogger("pipeline")


def run_stage(name: str, func: Callable, retries: int = 0, wait: float = 5.0):
    """Run one stage with timing; retry on I/O errors (e.g. a flaky download)."""
    for attempt in range(retries + 1):
        start = time.perf_counter()
        log.info("START %s", name)
        try:
            result = func()
        except OSError:
            log.exception("FAILED %s (attempt %d/%d)", name, attempt + 1, retries + 1)
            if attempt == retries:
                raise
            time.sleep(wait)
        else:
            log.info("DONE  %s (%.1fs)", name, time.perf_counter() - start)
            return result


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_pipeline(cfg: dict) -> int:
    """Run the whole workflow; return 0 on success, 1 if validation fails."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    if not raw_dir.exists():
        from plant_health.data.download import download

        run_stage("download", lambda: download(str(raw_dir)), retries=3)

    manifest = run_stage("manifest", lambda: build_manifest(raw_dir))

    report = run_stage("validate", lambda: validate(manifest, cfg))
    write_report(report, reports_dir)
    if not report["passed"]:
        for issue in report["issues"]:
            log.error("Validation: %s (%d) %s", issue["check"], issue["count"],
                      issue["examples"][:2])
        log.error("Validation failed, training aborted")
        return 1

    result = run_stage("train", lambda: train_run(cfg, manifest))

    stats = build_reference_stats(manifest)
    stats["confidence_validation"] = result["val_confidence"]
    _write_json(reports_dir / "reference_stats.json", stats)

    _write_json(
        reports_dir / "pipeline_run.json",
        {
            "config_name": cfg["name"],
            "run_id": result["run_id"],
            "model_dir": result["model_dir"],
            "metrics": {k: v for k, v in result.items() if k.startswith("final_val")},
        },
    )
    log.info("Pipeline finished: model saved to %s", result["model_dir"])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.exit(run_pipeline(load_config(args.config)))


if __name__ == "__main__":
    main()