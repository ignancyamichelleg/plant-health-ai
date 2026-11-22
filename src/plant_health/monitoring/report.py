"""Monitoring report: the request log cut into consecutive windows.

Writes reports/monitoring_windows.csv (one row per window) and
reports/monitoring_report.html (summary, charts, alerts, response plan).

    python -m plant_health.monitoring.report
"""
import argparse
import html
from pathlib import Path

import pandas as pd

from plant_health.monitoring.alerts import ACTIONS, check_window, window_stats
from plant_health.monitoring.logs import read_requests
from plant_health.monitoring.reference import load_reference
from plant_health.monitoring.settings import load_monitoring_config


def windows_table(requests: pd.DataFrame, cfg: dict, reference: dict | None) -> pd.DataFrame:
    """One row per consecutive window of `window.size` requests."""
    size = cfg["window"]["size"]
    rows = []
    for i, start in enumerate(range(0, len(requests), size)):
        chunk = requests.iloc[start : start + size]
        stats = window_stats(chunk)
        stats.pop("class_share", None)
        alerts = check_window(chunk, cfg, reference)
        stats.update(window=i + 1, alerts=";".join(a.rule for a in alerts))
        rows.append(stats)
    columns = [
        "window", "n", "start", "end", "low_quality_rate", "uncertain_rate",
        "brightness_mean", "blur_median", "confidence_mean", "latency_p95_ms", "alerts",
    ]
    return pd.DataFrame(rows, columns=columns)


def _svg_line(values: list[float], threshold: float | None, title: str) -> str:
    """Small dependency-free line chart, so the report opens anywhere."""
    w, h, pad = 420, 140, 28
    if not values:
        return ""
    top = max(values + ([threshold] if threshold is not None else [])) or 1.0
    top *= 1.1

    def xy(i, v):
        x = pad + (w - 2 * pad) * (i / max(len(values) - 1, 1))
        y = h - pad - (h - 2 * pad) * (v / top)
        return x, y

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, v) for i, v in enumerate(values)))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"/>'
        for x, y in (xy(i, v) for i, v in enumerate(values))
    )
    line = ""
    if threshold is not None:
        _, ty = xy(0, threshold)
        line = (
            f'<line x1="{pad}" x2="{w - pad}" y1="{ty:.1f}" y2="{ty:.1f}" '
            f'class="thr"/><text x="{w - pad}" y="{ty - 4:.1f}" '
            f'text-anchor="end" class="lbl">threshold {threshold:g}</text>'
        )
    return (
        f'<figure><figcaption>{html.escape(title)}</figcaption>'
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<line x1="{pad}" x2="{pad}" y1="{pad}" y2="{h - pad}" class="ax"/>'
        f'<line x1="{pad}" x2="{w - pad}" y1="{h - pad}" y2="{h - pad}" class="ax"/>'
        f'{line}<polyline points="{pts}" class="ser"/>{dots}'
        f'<text x="{pad}" y="{h - 8}" class="lbl">window 1</text>'
        f'<text x="{w - pad}" y="{h - 8}" text-anchor="end" class="lbl">'
        f"window {len(values)}</text></svg></figure>"
    )


def render_html(table: pd.DataFrame, latest_alerts: list, reference: dict | None,
                cfg: dict, total: int) -> str:
    a = cfg["alerts"]
    charts = "".join([
        _svg_line(table["low_quality_rate"].tolist(), a["low_quality_rate_max"],
                  "Low-quality rate per window"),
        _svg_line(table["brightness_mean"].tolist(), None, "Mean brightness per window"),
        _svg_line(table["confidence_mean"].tolist(), None, "Mean confidence per window"),
        _svg_line(table["latency_p95_ms"].tolist(), a["latency_p95_ms_max"],
                  "p95 latency (ms) per window"),
    ])
    if latest_alerts:
        alert_html = "<ul>" + "".join(
            f"<li><b>{html.escape(x.rule)}</b>: {html.escape(x.message)}<br>"
            f"<i>Action:</i> {html.escape(x.action)}</li>"
            for x in latest_alerts
        ) + "</ul>"
    else:
        alert_html = "<p>No alerts in the latest window.</p>"
    ref_html = (
        f"<p>Reference: {html.escape(reference['source'])} ({reference['n']} images), "
        f"mean brightness {reference['brightness']['mean']:.1f}, "
        f"median blur {reference['blur']['median']:.1f}.</p>"
        if reference else
        "<p>No reference stats found; drift rules were skipped.</p>"
    )
    plan = "".join(
        f"<tr><td>{html.escape(rule)}</td><td>{html.escape(action)}</td></tr>"
        for rule, action in ACTIONS.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>PlantHealth AI monitoring report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;margin:1rem 0}}
td,th{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}}
tr.alert td{{background:#fdecea}}
figure{{display:inline-block;margin:.5rem}} figcaption{{font-size:.85rem;font-weight:600}}
.ax{{stroke:#999}} .thr{{stroke:#c62828;stroke-dasharray:4 3}}
.ser{{fill:none;stroke:#2e7d32;stroke-width:2}} circle{{fill:#2e7d32}}
.lbl{{font-size:10px;fill:#666}}
</style></head><body>
<h1>PlantHealth AI monitoring report</h1>
<p>{total} logged requests in windows of {cfg['window']['size']}.
Images are never stored; only quality, confidence and latency are logged.</p>
{ref_html}
<h2>Latest window</h2>{alert_html}
<h2>Trends</h2>{charts}
<h2>Windows</h2>
{_table_html(table)}
<h2>Response plan</h2>
<table><tr><th>Rule</th><th>Action</th></tr>{plan}</table>
<p>Labels are not available in production, so all rules use proxy signals:
input quality, confidence, predicted-class mix and latency.</p>
</body></html>"""


def _table_html(table: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in table.columns)
    rows = []
    for _, r in table.iterrows():
        cls = ' class="alert"' if r["alerts"] else ""
        cells = "".join(
            f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>"
            for v in r.tolist()
        )
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def build_report(cfg: dict) -> pd.DataFrame:
    paths = cfg["paths"]
    requests = read_requests(paths["request_log"])
    reference = load_reference(paths["reference_stats"])
    table = windows_table(requests, cfg, reference)
    latest = requests.tail(cfg["window"]["size"])
    latest_alerts = check_window(latest, cfg, reference)

    csv_path, html_path = Path(paths["report_csv"]), Path(paths["report_html"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    html_path.write_text(
        render_html(table, latest_alerts, reference, cfg, len(requests)), encoding="utf-8"
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_monitoring_config(args.config)
    table = build_report(cfg)
    print(f"{len(table)} windows -> {cfg['paths']['report_html']}")


if __name__ == "__main__":
    main()
