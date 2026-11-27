"""Read logs/requests.jsonl (written by serving/request_logger.py) into a table."""
import json
from pathlib import Path

import pandas as pd

LOG_COLUMNS = [
    "ts",
    "request_id",
    "model_version",
    "status",
    "predicted_class",
    "confidence",
    "latency_ms",
    "brightness",
    "blur",
    "width",
    "height",
]


def read_requests(path: str | Path) -> pd.DataFrame:
    """Return one row per logged request, oldest first.

    Malformed lines are skipped (a crash mid-write must not break monitoring);
    the number skipped is stored in df.attrs["skipped_lines"].
    """
    path = Path(path)
    records, skipped = [], 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not all(col in record for col in LOG_COLUMNS):
                skipped += 1
                continue
            records.append(record)

    df = pd.DataFrame(records, columns=LOG_COLUMNS)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.attrs["skipped_lines"] = skipped
    return df


def last_window(df: pd.DataFrame, size: int) -> pd.DataFrame:
    return df.tail(size).reset_index(drop=True)
