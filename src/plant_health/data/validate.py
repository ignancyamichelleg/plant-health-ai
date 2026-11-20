"""Validate the image manifest before training. Exits with code 1 if a check fails."""
import json
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from plant_health.config import load_config
from plant_health.data.manifest import build_manifest

REPORT_PATH = Path("reports/validation_report.json")
SPLITS = ["train", "validation", "test"]


def build_schema(classes: list[str], rules: dict) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "path": pa.Column(str, unique=True),
            "split": pa.Column(str, pa.Check.isin(SPLITS)),
            "label": pa.Column(str, pa.Check.isin(classes)),
            "file_size": pa.Column(int, pa.Check.gt(0)),
            "sha256": pa.Column(str),
            "width": pa.Column(int, pa.Check.ge(rules["min_side"])),
            "height": pa.Column(int, pa.Check.ge(rules["min_side"])),
            "channels": pa.Column(int, pa.Check.eq(3)),
            "brightness": pa.Column(
                float,
                pa.Check.in_range(rules["brightness_min"], rules["brightness_max"]),
            ),
            "blur": pa.Column(float, pa.Check.ge(rules["blur_min"])),
        },
        strict=False,
    )


def check_duplicates(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return (duplicates across splits, duplicates within one split)."""
    across, within = [], []
    for sha, group in df.groupby("sha256"):
        if len(group) < 2:
            continue
        entry = {"sha256": sha[:12], "paths": group["path"].tolist()}
        (across if group["split"].nunique() > 1 else within).append(entry)
    return across, within


def check_class_balance(
    df: pd.DataFrame, classes: list[str], tolerance: float
) -> list[str]:
    problems = []
    expected = 1 / len(classes)
    shares = df.groupby("split")["label"].value_counts(normalize=True)
    for split in df["split"].unique():
        for label in classes:
            share = float(shares.get((split, label), 0.0))
            if abs(share - expected) > tolerance:
                problems.append(f"{split}/{label}: share {share:.2f}")
    return problems


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    rules, classes = cfg["validation"], cfg["classes"]
    issues: list[dict] = []
    warnings: list[dict] = []

    # 1. every file must decode
    unreadable = df.loc[~df["readable"], "path"].tolist()
    if unreadable:
        issues.append(
            {"check": "readable_images", "count": len(unreadable),
             "examples": unreadable[:5]}
        )

    # 2. schema + domain/statistical rules on the readable images
    ok = df[df["readable"]].astype({"width": int, "height": int, "channels": int})
    try:
        build_schema(classes, rules).validate(ok, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.groupby(["column", "check"], dropna=False)
        for (column, check), group in failures:
            rows = group["index"].dropna().astype(int)
            examples = [ok.loc[i, "path"] for i in rows][:5]
            issues.append(
                {"check": f"{column}: {check}", "count": len(group),
                 "examples": examples}
            )

    # 3. leakage: identical images in different splits
    across, within = check_duplicates(ok)
    if across:
        issues.append(
            {"check": "no_duplicates_across_splits", "count": len(across),
             "examples": [", ".join(d["paths"]) for d in across[:3]]}
        )
    if within:
        warnings.append(
            {"check": "duplicates_within_split", "count": len(within),
             "examples": [", ".join(d["paths"]) for d in within[:3]]}
        )

    # 4. class balance per split
    imbalance = check_class_balance(ok, classes, rules["class_share_tolerance"])
    if imbalance:
        issues.append(
            {"check": "class_balance", "count": len(imbalance),
             "examples": imbalance[:5]}
        )

    return {
        "passed": not issues,
        "n_images": len(df),
        "n_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "issues": issues,
        "warnings": warnings,
        "rules": rules,
        "observed": {
            "brightness_min": float(ok["brightness"].min()),
            "brightness_max": float(ok["brightness"].max()),
            "blur_min": float(ok["blur"].min()),
        },
    }


def main() -> int:
    cfg = load_config()
    report = validate(build_manifest(cfg["data"]["raw_dir"]), cfg)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Validation {status}: {report['n_images']} images")
    for issue in report["issues"]:
        print(f"  FAIL {issue['check']} ({issue['count']}): {issue['examples'][:2]}")
    for warning in report["warnings"]:
        print(f"  WARN {warning['check']} ({warning['count']})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())