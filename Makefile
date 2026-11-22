CONFIG ?= configs/experiments/baseline_cnn.yaml

.PHONY: setup data validate train test lint mlflow-ui

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

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db