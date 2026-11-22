param(
  [Parameter(Position = 0)][string]$Task = "help",
  [string]$Config = "configs/experiments/baseline_cnn.yaml"
)

switch ($Task) {
  "setup"     { pip install -e ".[dev]" }
  "data"      { python -m plant_health.data.download }
  "validate"  { python -m plant_health.data.validate }
  "train"     { python pipelines/train_pipeline.py --config $Config }
  "test"      { pytest -q }
  "lint"      { ruff check . }
  "mlflow-ui" { mlflow ui --backend-store-uri sqlite:///mlflow.db }
  default     { Write-Host "Tasks: setup data validate train test lint mlflow-ui" }
}