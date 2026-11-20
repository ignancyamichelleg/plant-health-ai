"""FastAPI application for PlantHealth AI inference."""

import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from plant_health.monitoring.quality import image_quality
from plant_health.serving.predictor import Predictor
from plant_health.serving.request_logger import log_prediction
from plant_health.serving.schemas import HealthResponse, PredictionResponse

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
CONTENT_TYPE_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
# Provisional thresholds; calibrate with Person 3.
CONFIDENCE_THRESHOLD = 0.60
BRIGHTNESS_MIN = 30.0
BRIGHTNESS_MAX = 230.0
BLUR_MIN = 50.0


def create_app(predictor=None):
    """Create an application with an optional injected predictor for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            model_dir = Path(
                os.getenv("PLANT_HEALTH_MODEL_DIR", "models/best")
            )
            app.state.predictor = Predictor(model_dir)

        yield

    app = FastAPI(
        title="PlantHealth AI",
        description="Bean leaf image classification prototype.",
        version="0.1.0",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parent / "static"

    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(static_dir / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_version=app.state.predictor.model_version,
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: Annotated[UploadFile, File()]):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())

        if file.content_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image content type",
            )

        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 8 MB upload limit",
            )

        try:
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image format",
                    )
                if image.format != CONTENT_TYPE_TO_FORMAT[file.content_type]:
                    raise HTTPException(
                        status_code=415,
                        detail="Image content type does not match actual format",
                    )
                image.load()
                width, height = image.size
                quality = image_quality(image)

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image",
            ) from exc

        result = app.state.predictor.predict(contents)

        if (
            quality["brightness"] < BRIGHTNESS_MIN
            or quality["brightness"] > BRIGHTNESS_MAX
            or quality["blur"] < BLUR_MIN
        ):
            status = "low_quality"
        elif result["confidence"] < CONFIDENCE_THRESHOLD:
            status = "uncertain"
        else:
            status = "ok"

        latency_ms = (time.perf_counter() - started) * 1000

        log_prediction(
            request_id=request_id,
            model_version=result["model_version"],
            status=status,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
            brightness=quality["brightness"],
            blur=quality["blur"],
            width=width,
            height=height,
        )

        return PredictionResponse(
            request_id=request_id,
            status=status,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            quality=quality,
            model_version=result["model_version"],
            latency_ms=latency_ms,
        )

    return app


app = create_app()
