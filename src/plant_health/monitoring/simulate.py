"""Simulated production incident: normal traffic, then dark/blurry photos.

Sends real images to the running API (POST /predict), so the requests go
through the same quality checks and logging as phone traffic. Afterwards it
runs the alert check and builds the monitoring report.

    make serve                          # in one terminal
    python -m plant_health.monitoring.simulate --images data/raw/validation
"""
import argparse
import io
import json
import random
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from plant_health.monitoring.alerts import run as run_alerts
from plant_health.monitoring.report import build_report
from plant_health.monitoring.settings import load_monitoring_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def degrade(img: Image.Image, mode: str) -> Image.Image:
    """Return a degraded copy: 'dark', 'blur' or 'both'."""
    img = img.convert("RGB")
    if mode in ("dark", "both"):
        img = ImageEnhance.Brightness(img).enhance(0.15)
    if mode in ("blur", "both"):
        img = img.filter(ImageFilter.GaussianBlur(radius=8))
    return img


def to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def multipart(image_bytes: bytes, filename: str = "leaf.jpg") -> tuple[bytes, str]:
    """Encode one file field as multipart/form-data (stdlib only)."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def http_poster(base_url: str) -> Callable[[bytes], dict]:
    url = base_url.rstrip("/") + "/predict"

    def post(image_bytes: bytes) -> dict:
        body, content_type = multipart(image_bytes)
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    return post


def find_images(folder: str | Path) -> list[Path]:
    paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"No images found under {folder}")
    return sorted(paths)


def simulate(
    images: list[Path],
    post: Callable[[bytes], dict],
    normal: int,
    degraded: int,
    mode: str,
    seed: int = 42,
    pause_s: float = 0.0,
) -> list[dict]:
    """Send `normal` clean images, then `degraded` degraded ones. Returns responses."""
    rng = random.Random(seed)
    responses = []
    for phase, count in (("normal", normal), (mode, degraded)):
        for _ in range(count):
            with Image.open(rng.choice(images)) as img:
                img.load()
                if phase != "normal":
                    img = degrade(img, mode)
                responses.append({"phase": phase, **post(to_jpeg(img))})
            if pause_s:
                time.sleep(pause_s)
    return responses


def summarize(responses: list[dict]) -> dict:
    out: dict = {}
    for r in responses:
        phase = out.setdefault(r["phase"], {})
        phase[r["status"]] = phase.get(r["status"], 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", default="data/raw/validation")
    parser.add_argument("--normal", type=int, default=50)
    parser.add_argument("--degraded", type=int, default=50)
    parser.add_argument("--mode", choices=["dark", "blur", "both"], default="dark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    responses = simulate(
        find_images(args.images), http_poster(args.url),
        args.normal, args.degraded, args.mode, args.seed,
    )
    print("Responses by phase:", json.dumps(summarize(responses)))

    cfg = load_monitoring_config(args.config)
    result = run_alerts(cfg)
    build_report(cfg)
    for alert in result["alerts"]:
        print(f"ALERT [{alert['rule']}] {alert['message']}")
    if not result["alerts"]:
        print("No alerts fired - check thresholds in configs/monitoring.yaml")
    print(f"Report: {cfg['paths']['report_html']}")


if __name__ == "__main__":
    main()
