FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLANT_HEALTH_MODEL_DIR=/app/models/best

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/models/best /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "plant_health.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
