CONFIG ?= configs/experiments/baseline_cnn.yaml
URL ?= http://localhost:8000
MODE ?= dark

.PHONY: setup data validate train test lint check serve reference monitor report simulate mlflow-ui

setup:
	pip install -e ".[dev]"

data:
	python -m plant_health.data.download

validate:
	python -m plant_health.data.validate

train:
	python pipelines/train_pipeline.py --config $(CONFIG)

test:
	pytest -q

lint:
	ruff check .

check: lint test

serve:
	uvicorn plant_health.serving.api:app --host 0.0.0.0 --port 8000

# Monitoring
reference:
	python -m plant_health.monitoring.reference

monitor:
	python -m plant_health.monitoring.alerts

report:
	python -m plant_health.monitoring.report

simulate:
	python -m plant_health.monitoring.simulate --url $(URL) --mode $(MODE)

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
