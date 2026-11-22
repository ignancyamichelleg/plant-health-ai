"""Reference statistics that live traffic is compared against.

Built from the validation split of data/manifest.csv (brightness, blur and
class mix). Mean confidence is added when available (e.g. from evaluate.py),
otherwise the confidence-drop alert is skipped.

    python -m plant_health.monitoring.reference
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from plant_health.monitoring.settings import load_monitoring_config

MANIFEST_PATH = Path("data/manifest.csv")


def _describe(series: pd.Series) -> dict:
    s = series.astype(float)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
    }


def build_reference(
    manifest: pd.DataFrame,
    split: str = "validation",
    confidences: list[float] | None = None,
) -> dict:
    rows = manifest[(manifest["split"] == split) & (manifest["readable"])]
    if rows.empty:
        raise ValueError(f"No readable rows for split '{split}' in the manifest")
    shares = rows["label"].value_counts(normalize=True).sort_index()
    return {
        "source": f"manifest:{split}",
        "n": len(rows),
        "brightness": _describe(rows["brightness"]),
        "blur": _describe(rows["blur"]),
        "class_share": {k: float(v) for k, v in shares.items()},
        "confidence": (
            {"mean": float(pd.Series(confidences).mean())} if confidences else None
        ),
    }


def load_reference(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_monitoring_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", default=cfg["paths"]["reference_stats"])
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    reference = build_reference(manifest, split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"Wrote reference stats ({reference['n']} images) to {out}")


if __name__ == "__main__":
    main()
