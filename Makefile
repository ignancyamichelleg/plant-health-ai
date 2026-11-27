CONFIG ?= configs/experiments/baseline_cnn.yaml
URL ?= http://localhost:8000
MODE ?= dark
MODEL ?= models/baseline_cnn

.PHONY: setup data validate train test lint check serve evaluate select monitor report simulate mlflow-ui

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

evaluate:
	python -m plant_health.models.evaluate --model $(MODEL)

select:
	python -m plant_health.models.evaluate --select

serve:
	uvicorn plant_health.serving.api:app --host 0.0.0.0 --port 8000

# Monitoring (reports/reference_stats.json is written by `make train`)
monitor:
	python -m plant_health.monitoring.alerts

report:
	python -m plant_health.monitoring.report

simulate:
	python -m plant_health.monitoring.simulate --url $(URL) --mode $(MODE)

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
