# PlantHealth AI

**DTI6302 — Machine Learning Operations**

PlantHealth AI is an end-to-end machine learning system
for classifying bean leaf images and demonstrating the
operational lifecycle of a machine learning application.

The project integrates dataset validation, reproducible
training, experiment tracking, model selection, inference
serving, containerization, automated testing and
operational monitoring.

## Table of contents

- [1. Project overview](#1-project-overview)
- [2. System architecture](#2-system-architecture)
- [3. Dataset](#3-dataset)
- [4. Installation](#4-installation)
- [5. Training and experiment tracking](#5-training-and-experiment-tracking)
- [6. Model evaluation and selection](#6-model-evaluation-and-selection)
- [7. Model artifacts and provenance](#7-model-artifacts-and-provenance)
- [8. Native application deployment](#8-native-application-deployment)
- [9. Docker deployment](#9-docker-deployment)
- [10. Monitoring and reliability](#10-monitoring-and-reliability)
- [11. Automated testing and continuous integration](#11-automated-testing-and-continuous-integration)
- [12. Makefile commands](#12-makefile-commands)
- [13. Repository structure](#13-repository-structure)
- [14. Limitations and future work](#14-limitations-and-future-work)
- [15. Team contributions](#15-team-contributions)
- [16. Resources](#16-resources)

## 1. Project overview

The application classifies bean leaf images into three
categories:

- Angular leaf spot
- Bean rust
- Healthy

Users can submit images through a responsive web interface
and receive a predicted class, confidence score, class
probabilities and image-quality feedback.

The system is an educational prototype and is not intended
to replace professional agricultural diagnosis.

### Problem definition and intended users

The practical problem is supporting preliminary identification
of common bean leaf diseases through accessible image-based
screening. Potential users include growers, agricultural field
workers and agricultural students. These are intended user
groups, not populations with whom formal user testing has
been conducted.

The machine learning task is supervised three-class image
classification. The unit of prediction is one submitted bean
leaf image. The input is an RGB image transformed into a
160 x 160 x 3 tensor, and the target is one of the three
predefined categories.

An image-based model is appropriate because the target
categories are associated with visible leaf characteristics.
A custom CNN provides a practical baseline against which
augmentation and transfer learning can be evaluated.

### Evaluation and operational success criteria

Predictive performance is assessed primarily using validation
macro-F1, complemented by accuracy and per-class recall.

Operational acceptance is assessed through:

- Successful inference using the documented API contract.
- Consistent predictions across tested deployment environments.
- Shared preprocessing between training and inference.
- Explicit feedback for low-quality and uncertain inputs.
- Detection of deliberately degraded input traffic.
- Passing automated tests and static code checks.

The monitoring configuration defines a P95 latency alert
threshold of 1,000 ms. This is an operational threshold,
not a production service-level guarantee.

These criteria summarize the retrospective evaluation
framework used to assess the prototype. No prospective
minimum macro-F1 acceptance threshold or formal user
satisfaction target was established.

## 2. System architecture

![PlantHealth AI architecture](docs/architecture.svg)

The architecture comprises four interconnected layers:

1. **Data and validation:** dataset acquisition, manifest
   generation, quality assessment and leakage checks.
2. **Training and evaluation:** configurable experiments,
   shared preprocessing, MLflow tracking, model registration
   and validation-based model selection.
3. **Deployment:** FastAPI inference, responsive web
   interface and Docker containerization.
4. **Monitoring:** request telemetry, quality checks,
   distribution-shift alerts, operational reports and
   controlled incident simulation.

## 3. Dataset

The project uses the Beans dataset published by Makerere
AI Lab on Hugging Face.

- Source: https://huggingface.co/datasets/AI-Lab-Makerere/beans
- License: MIT
- Original format: 500 x 500 RGB JPEG images
- Model input size: 160 x 160 RGB

| Split | Images |
|---|---:|
| Training | 1,034 |
| Validation | 133 |
| Test | 128 |
| Total | 1,295 |

The dataset validation process checks image integrity,
class labels, schema consistency, image quality and
cross-split duplication.

See [DATA.md](DATA.md) for additional dataset information.

### Dataset methodology and limitations

The original dataset splits are preserved. Training data are
used for fitting model parameters, validation data for model
comparison and selection, and test data for final held-out
assessment.

The validation pipeline checks data integrity, class
consistency, image quality and cross-split duplication
to reduce data leakage risk.

The verified dataset validation report covers all 1,295
images and records no validation issues or warnings.

Performance may differ under unfamiliar lighting, backgrounds,
camera conditions, disease severity or geographic environments.
Other plant species, diseases and mixed conditions are outside
the defined classification task.

## 4. Installation

### Requirements

- Python 3.10–3.12; Python 3.11 recommended
- Git
- Docker, for containerized deployment
- Internet access for initial dependency installation,
  dataset acquisition and pretrained weights

The project was locally validated using Python 3.11.18
on macOS with Apple Silicon.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ignancyamichelleg/plant-health-ai.git
cd plant-health-ai
```

Until the final team integration into the default branch
is completed, switch to the integrated project branch:

```bash
git switch p2/mobilenet-and-serving
```

Create and activate a Python virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Download and validate the dataset:

```bash
python -m plant_health.data.download
python -m plant_health.data.validate
```

The dataset is stored locally under:

```text
data/raw/
├── train/
├── validation/
└── test/
```

### Independent reproducibility verification

An independent checkout of commit
`55067f43b20b1bbff4549a7a634800ee7b00410c`
was used to verify installation and execution.

| Verification | Observed result |
|---|---|
| Fresh Python virtual environment | Installation successful |
| Dependency consistency | No broken requirements |
| Required package imports | 13/13 passed |
| Static code checks | Passed |
| Automated tests at that revision | 78/78 passed |
| Independently downloaded dataset | 1,295 images |
| Dataset file comparison | 1,295/1,295 byte-identical |
| Dataset validation report | Matched the official report |
| Independent native deployment | Successful |
| Cross-environment inference checks | 74/74 passed |

Verification used Python 3.11.18 on macOS with Apple Silicon.
Dependencies were installed into a new virtual environment
rather than copied from the original environment.

These results establish reproducibility of the tested
installation, dataset acquisition and inference workflow
at the recorded revision. They do not establish deterministic
retraining or guarantee identical results with future
dependency versions and upstream dataset revisions.

## 5. Training and experiment tracking

Three candidate architectures are implemented:

| Experiment | Architecture | Augmentation |
|---|---|---|
| baseline_cnn | Custom CNN | No |
| cnn_aug | Custom CNN | Yes |
| mobilenet_frozen | Frozen ImageNet MobileNetV2 backbone | Yes |

The experiments use shared preprocessing and configurable
training parameters.

The base configuration specifies a random seed of 42,
an image size of 160, a batch size of 32, a maximum of
60 epochs and an early stopping patience of 10.

Experiment-specific YAML files can override these settings.
The repository's MobileNetV2 configuration specifies a
learning rate of 0.0001, 20 maximum epochs and an early
stopping patience of 5.

The historical configuration at commit `519e2a8` specifies
60 maximum epochs and patience 10 for the two CNN experiments.
The MobileNetV2 experiment overrides these values with
20 maximum epochs and patience 5.

These are the effective committed configuration values.
Original MLflow records are required to independently confirm
the actual parameters used in each official training run.

### Run training experiments

```bash
python pipelines/train_pipeline.py \
  --config configs/experiments/baseline_cnn.yaml

python pipelines/train_pipeline.py \
  --config configs/experiments/cnn_aug.yaml

python pipelines/train_pipeline.py \
  --config configs/experiments/mobilenet_frozen.yaml
```

Training generates model artifacts, metadata, pipeline
reports and monitoring reference statistics.

MLflow records experiment parameters, metrics and artifacts.

Launch the local MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000.

Training commands regenerate local experiment artifacts
and may update existing reports. Preserve official results
before repeating experiments.

## 6. Model evaluation and selection

The candidate models are evaluated using validation
accuracy, macro-F1, per-class recall, confusion matrices
and image-quality slices.

Official validation results:

| Model | Accuracy | Macro-F1 | Minimum class recall |
|---|---:|---:|---:|
| baseline_cnn | 84.96% | 84.89% | 77.78% |
| cnn_aug | 82.71% | 82.41% | 65.91% |
| mobilenet_frozen | 81.20% | 80.85% | 63.64% |

**Selected model: baseline_cnn.**

The baseline CNN achieved the highest validation macro-F1,
accuracy and minimum per-class recall among the candidates.
Its macro-F1 exceeded the augmented CNN by approximately
2.48 percentage points and MobileNetV2 by 4.04 percentage points.

The selection is supported by observed validation results
rather than model complexity or architectural preference.
These experiments do not establish that augmentation or
transfer learning is inherently inferior.

Model selection prioritizes validation macro-F1, with
minimum per-class recall as the tie-breaking criterion.

The test split is excluded from model selection.

See [the official model selection report](reports/model_selection.md)
for the comparative evaluation.

To reproduce validation evaluation:

```bash
python -m plant_health.models.evaluate \
  --model models/baseline_cnn --split validation

python -m plant_health.models.evaluate \
  --model models/cnn_aug --split validation

python -m plant_health.models.evaluate \
  --model models/mobilenet_frozen --split validation

python -m plant_health.models.evaluate --select
```

The final held-out test evaluation is protected by an
explicit final-evaluation flag and a local evaluation
record. It should be performed only once on the selected
model as a coordinated team activity.

The validation results above must not be interpreted as
held-out test performance.

### Failure analysis

The selected baseline CNN achieved validation recall of
84.09% for angular leaf spot, 77.78% for bean rust and
93.18% for healthy leaves.

Bean rust therefore had the lowest recall for the selected
model. For frozen MobileNetV2, angular leaf spot was the
weakest class, with recall of 63.64%.

Per-class results, confusion matrices, misclassified examples
and image-quality slices are available in the evaluation
outputs under `reports/`.

These analyses identify potential failure modes but do not
establish causal relationships between image quality and
classification errors.

The final-evaluation guard is local to the evaluation
environment. It does not prevent repeated test evaluation
across independent machines or repository checkouts.

## 7. Model artifacts and provenance

Each trained model is stored with its serving metadata:

```text
models/
├── baseline_cnn/
│   ├── model.keras
│   └── metadata.json
├── cnn_aug/
│   ├── model.keras
│   └── metadata.json
└── mobilenet_frozen/
    ├── model.keras
    └── metadata.json
```

Model metadata includes the model version, class order,
input size, input specification and experiment provenance,
including the MLflow run ID, Git commit and data hash.

Trained weights are distributed separately through the
team's shared model artifacts or regenerated through
the training pipeline. They are not committed to Git.

For the official deployment, use the distributed
baseline_cnn model and its corresponding metadata.

### Official experiment traceability

| Model | MLflow run ID |
|---|---|
| baseline_cnn | `e1af6859d2dc4ceeb53bec024fe34892` |
| cnn_aug | `b1fbe83ba2e34849b8688d5081be7fb2` |
| mobilenet_frozen | `159d85e0b50d467e86f215e22acb1f6d` |

All three official model metadata records identify training
commit `519e2a8` and data hash `19e7a4b9a26b`.

The selected model artifact has SHA-256 fingerprint:

`e97342155daaa482d76b65da74b96de39752be6dd9179ee0499034bc5d85e0de`

Its corresponding metadata fingerprint is:

`f02eb4841a3ae7bd2e1f2aba8674b84070e00b22fd00bce47fccb4c616377c3f`

Official metadata and evaluation reports were reconciled
during integration. Original historical MLflow database
records and associated artifacts constitute separate
experiment evidence.

An independently executed MobileNetV2 development experiment
has run ID `1b9108ec1a8f45f588db0ac4f33a0087`.
It is distinct from the official MobileNetV2 experiment.

## 8. Native application deployment

From the repository root, with the virtual environment
activated:

```bash
PLANT_HEALTH_MODEL_DIR=models/baseline_cnn \
python -m uvicorn plant_health.serving.api:app \
  --host 127.0.0.1 --port 8000
```

Open the application:

- Web interface: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

The responsive web interface supports image submission,
image preview, prediction results, class probabilities
and quality or uncertainty feedback.

### Runtime configuration

The supported environment variables are documented in
`.env.example`.

| Variable | Default |
|---|---|
| `PLANT_HEALTH_MODEL_DIR` | `models/best` |
| `PLANT_HEALTH_MONITORING_CONFIG` | `configs/monitoring.yaml` |
| `PLANT_HEALTH_REQUEST_LOG` | `logs/requests.jsonl` |

Set these variables in the shell or container environment
before launching the application. The `.env.example` file
is a reference template and is not automatically loaded.

The Docker image sets its model directory to `/app/models/best`.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Web interface |
| GET | /health | Application and model status |
| POST | /predict | Image classification |
| GET | /docs | Interactive API documentation |

The prediction endpoint accepts multipart image uploads
using the field name `file`.

Supported image formats include JPEG, PNG and WebP.
The maximum uploaded file size is 8 MiB. Decoded image
dimensions are additionally limited to 25 million pixels.

The API checks supported content types and image formats,
handles corrupted images and rejects unsafe image dimensions.

Oversized files and images exceeding the decoded-image
limit receive an HTTP 413 response. Invalid and unsupported
images receive appropriate client-error responses.

The file-size check is applied within the prediction endpoint.
It is not an infrastructure-level HTTP request-body limit.

These controls reduce resource-exhaustion risks but do not
constitute comprehensive production security.

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/leaf.jpg"
```

The response contains:

- Request ID
- Prediction status
- Predicted class
- Confidence score
- Class probabilities
- Image-quality metrics
- Model version
- Inference latency

Prediction statuses are:

| Status | Meaning |
|---|---|
| ok | Quality and confidence thresholds satisfied |
| uncertain | Model confidence below the configured threshold |
| low_quality | Image fails one or more quality thresholds |

An `ok` status does not guarantee a correct classification.

### Recorded prediction response

The following response was captured during independent
deployment verification at commit `55067f4`. It is an actual
response rather than an illustrative reconstruction.

```json
{
  "request_id": "2738b023-c6fe-43f5-a24f-1d93ada2908e",
  "status": "uncertain",
  "predicted_class": "bean_rust",
  "confidence": 0.5094732046127319,
  "probabilities": {
    "angular_leaf_spot": 0.012794477865099907,
    "bean_rust": 0.5094732046127319,
    "healthy": 0.4777323305606842
  },
  "quality": {
    "brightness": 89.16285705566406,
    "blur": 3183.79638671875
  },
  "model_version": "baseline_cnn",
  "latency_ms": 34.384208265691996
}
```

The recorded latency is specific to this request and
environment, not a formal performance benchmark.

## 9. Docker deployment

The serving application is packaged using a Python 3.11
slim base image and runs as a non-root user.

Build the image from the repository root:

```bash
docker build -t plant-health-ai:final .
```

Ensure the selected model artifacts are available locally.

Create a persistent request-log directory:

```bash
mkdir -p logs/docker-final
```

Start the container:

```bash
docker run -d \
  --name plant-health-final \
  -p 127.0.0.1:8001:8000 \
  -e PLANT_HEALTH_MODEL_DIR=/app/models/best \
  -v "$(pwd)/models/baseline_cnn:/app/models/best:ro" \
  -v "$(pwd)/logs/docker-final:/app/logs" \
  plant-health-ai:final
```

The model directory is mounted read-only. Request logs
are persisted outside the container.

The Docker application is available at:

http://127.0.0.1:8001

Verify deployment:

```bash
curl http://127.0.0.1:8001/health

docker logs plant-health-final
```

Stop and remove the container when finished:

```bash
docker stop plant-health-final
docker rm plant-health-final
```

The application has been validated in native Python and
Docker environments. For three paired validation images,
both environments returned identical predicted classes,
statuses and probability distributions.

This establishes consistency for the tested inputs, not
general classification accuracy or exhaustive platform
equivalence.

### Training-serving consistency and deployment verification

Training and inference use shared image preprocessing.
Images are converted to RGB, resized using a centered crop
to 160 x 160 pixels and represented as float32 values
in the range 0-255. Model-specific scaling is incorporated
within the model architecture.

The predictor loads the saved model and corresponding
metadata, preserving the expected input specification
and class ordering.

Independent deployment verification extended the comparison
to three environments:

- Original native Python application.
- Docker-containerized application.
- Native application launched from a fresh repository checkout.

Three validation images were submitted to each environment,
producing nine actual prediction requests.

All 74 verification checks passed, including six exact
cross-environment probability comparisons.

Two of the three distinct images were correctly classified.
The remaining image received an uncertain prediction status.
This small demonstration is not an estimate of model accuracy.

The independent verification was conducted at commit
`55067f4`, before the subsequent API security improvements.
The official model artifacts were preserved unchanged
during those improvements.

## 10. Monitoring and reliability

The monitoring system operates at two levels.

### Individual request monitoring

Each prediction is assessed using:

| Metric | Threshold |
|---|---:|
| Minimum confidence | 0.60 |
| Minimum brightness | 30 |
| Maximum brightness | 230 |
| Minimum blur metric | 50 |

Image quality takes precedence over confidence when
determining the prediction status.

Prediction telemetry is stored in JSON Lines format.
The request logs contain operational metadata rather
than uploaded image files.

### Aggregate monitoring

The monitoring system evaluates windows of 50 requests,
with a minimum of 20 requests required for alerting.

| Alert | Threshold |
|---|---:|
| Low-quality request rate | Above 20% |
| Uncertain request rate | Above 30% |
| Mean brightness shift | Above 40 |
| Median blur ratio | Below 0.50 |
| Mean confidence drop | Above 0.15 |
| Predicted-class share shift | Above 0.25 |
| P95 latency | Above 1,000 ms |

Reference statistics are derived from training-image
quality measurements and validation confidence scores
for the selected model.

Monitoring configuration is defined in:

`configs/monitoring.yaml`

The Docker application uses matching built-in defaults
when the YAML file is unavailable.

For custom container settings, mount the configuration
file read-only and set `PLANT_HEALTH_MONITORING_CONFIG`
to its container path.

### Monitoring commands

For the default native request log:

```bash
python -m plant_health.monitoring.alerts
python -m plant_health.monitoring.report
```

Generated outputs include:

- `reports/alerts.json`
- `reports/monitoring_windows.csv`
- `reports/monitoring_report.html`

These commands can overwrite existing monitoring reports.
Use an independent configuration for new experiments
when official reports must be preserved.

For Docker request logs, provide a monitoring configuration
whose `paths.request_log` points to
`logs/docker-final/requests.jsonl`.

### Controlled incident simulation

With the native application running:

```bash
python -m plant_health.monitoring.simulate \
  --url http://127.0.0.1:8000 \
  --images data/raw/validation \
  --normal 50 \
  --degraded 50 \
  --mode dark \
  --seed 42
```

The simulator sends normal and degraded images through
the actual prediction endpoint.

An isolated Docker integration experiment produced the
following results:

| Metric | Normal | Degraded |
|---|---:|---:|
| Requests | 50 | 50 |
| Low-quality rate | 0% | 100% |
| Uncertain rate | 8% | 0% |
| Mean confidence | 83.15% | 99.66% |
| Mean brightness | 123.47 | 18.32 |
| Median blur metric | 916.91 | 21.86 |

Four operational alerts were triggered:

- Low-quality request rate
- Brightness shift
- Blur degradation
- Predicted-class distribution shift

All degraded inputs were flagged as low quality despite
the increase in model confidence.

This experiment demonstrates the importance of input
quality monitoring alongside confidence-based checks.
It does not establish real-world drift detection accuracy.

The repository also contains the team's official
monitoring reports.

### Operational incident response

During the controlled degradation experiment, mean model
confidence increased by 16.51 percentage points while all
50 degraded inputs were flagged as low quality.

All 50 degraded inputs were classified as bean rust.
This distribution shift is an operational proxy signal,
not evidence of changing disease prevalence or a measured
reduction in predictive accuracy.

Recommended operational response:

| Signal | Recommended action |
|---|---|
| High low-quality rate | Investigate image acquisition and request clearer images |
| Brightness or blur shift | Review lighting, focus and camera conditions |
| High uncertainty rate | Investigate unfamiliar or ambiguous inputs |
| Confidence degradation | Review input distributions and collect labeled examples |
| Predicted-class share shift | Investigate possible input or population changes |
| Excessive latency | Investigate resource utilization and service performance |

Following mitigation, evaluate subsequent monitoring windows
to determine whether the affected signals return to acceptable
levels.

If abnormalities persist, obtain representative labeled
examples and reassess model performance before considering
retraining.

When ground-truth labels become available, compare predictions
with confirmed labels to calculate actual performance metrics,
including macro-F1, accuracy and per-class recall.

Until labels are available, monitoring signals indicate
potential problems rather than measured predictive accuracy.

The controlled experiment verified incident detection.
A separate post-incident recovery window was not evaluated.

## 11. Automated testing and continuous integration

Run the complete automated test suite:

```bash
python -m pytest -q
```

Run static code checks:

```bash
ruff check .
```

The integrated implementation at commit `3229238`
passed 78 automated tests locally, with no failures.

Following the API security improvements, the updated
implementation passed all 82 automated tests.

| Verification | Result |
|---|---|
| Complete automated test suite | 82/82 passed |
| API-specific tests | 14/14 passed |
| Failures, errors and skipped tests | 0 |
| Full test execution time | 11.28 seconds |
| Ruff static code checks | Passed |
| Operational request logs | Unchanged |
| Official model artifacts | Unchanged |
| Official evaluation reports | Unchanged |

The additional API tests cover decoded-image size rejection,
Pillow decompression error handling, uncertainty independently
of image quality, and successful confident prediction.

API tests use isolated request logs to prevent automated
testing from contaminating operational monitoring evidence.

The full regression run produced 287 third-party dependency
deprecation warnings, consistent with the earlier baseline.

The test suite covers data validation, preprocessing,
model architecture, training, evaluation, prediction
serving, request logging, monitoring and pipeline
integration.

The GitHub Actions workflow executes:

1. Dependency installation using Python 3.11.
2. Ruff static code checks.
3. Unit and integration tests.
4. Docker image build after successful checks.

The workflow is triggered by pull requests and pushes
to the main branch.

The workflow configuration is present in the repository.
Successful local testing does not establish successful
hosted GitHub Actions execution. Remote CI results must
be verified against the final integrated revision.

## 12. Makefile commands

| Command | Purpose |
|---|---|
| make setup | Install project and development dependencies |
| make data | Download the dataset |
| make validate | Validate dataset integrity |
| make train | Train a configured model |
| make evaluate | Evaluate a saved model |
| make select | Select the best validation model |
| make test | Run automated tests |
| make lint | Run static code checks |
| make check | Run linting and tests |
| make serve | Start the inference API |
| make monitor | Evaluate operational alerts |
| make report | Generate monitoring reports |
| make simulate | Simulate degraded input traffic |
| make mlflow-ui | Start MLflow |

Example custom experiment:

```bash
make train CONFIG=configs/experiments/mobilenet_frozen.yaml
```

Example selected-model serving:

```bash
PLANT_HEALTH_MODEL_DIR=models/baseline_cnn make serve
```

Note that `make serve` binds to all network interfaces.
Use the explicit localhost deployment command when
network exposure is not required.

## 13. Repository structure

```text
plant-health-ai/
├── .github/workflows/     # Continuous integration
├── configs/               # Training and monitoring settings
├── docs/                  # System architecture
├── pipelines/             # Training pipeline entry point
├── reports/               # Evaluation and monitoring outputs
├── src/plant_health/
│   ├── data/              # Dataset acquisition and validation
│   ├── features/          # Shared image preprocessing
│   ├── models/            # Training, evaluation and registry
│   ├── serving/           # API, predictor and web interface
│   └── monitoring/        # Quality metrics and alerts
├── tests/                 # Unit and integration tests
├── DATA.md
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## 14. Limitations and future work

- The model supports only three predefined bean leaf
  categories and is not a general plant disease detector.
- Model confidence does not guarantee prediction accuracy.
- Quality thresholds cannot identify every unreliable or
  out-of-distribution input.
- Monitoring relies on proxy signals when ground-truth
  labels are unavailable during inference.
- The incident simulation uses artificial degradation
  rather than actual production traffic.
- Local latency measurements are not formal load tests.
- The application has been validated locally, not as a
  publicly hosted production service.
- Dependency deprecation warnings remain a maintenance
  consideration.
- Fully pinned dependencies, broader external validation,
  automated monitoring scheduling and production security
  controls are potential future improvements.

### Responsible use and deployment security

Incorrect predictions could lead to inappropriate agricultural
decisions. Results should be treated as preliminary screening
information rather than definitive diagnoses.

The system does not include a dedicated out-of-distribution
detector or a calibrated guarantee of prediction correctness.
Quality warnings and confidence thresholds are safeguards,
not substitutes for expert assessment.

The documented local deployment uses localhost port bindings,
non-root container execution, read-only model mounts, file
validation and decoded-image limits.

Application prediction logs store operational metadata
rather than uploaded image files. This does not constitute
a comprehensive privacy or security assessment.

Public production deployment would require additional
controls, including authentication, TLS, infrastructure-level
request limits, rate limiting, dependency vulnerability
management, log retention policies and resource isolation.

Broader field validation, delayed-label performance assessment,
pinned dependency versions and a pinned dataset revision
would strengthen reliability and reproducibility.

## 15. Team contributions

The project was developed collaboratively through three
complementary workstreams:

| Workstream | Primary responsibilities |
|---|---|
| Person 1 - Ignancya Michelle George | Dataset acquisition, validation, training pipeline, experiment tracking and official training experiments |
| Person 2 - Juan Sebastian Ramirez Acuña | MobileNetV2 implementation, inference predictor, FastAPI, responsive web interface, Docker deployment, integration validation, reproducibility verification, API security improvements and integrated documentation |
| Person 3 - Ferdous Pathan | Model evaluation, monitoring configuration, operational alerts, reporting, incident simulation and CI integration |

The final implementation integrates contributions across
all three workstreams.

## 16. Resources

### Academic attribution and external resources

The project uses the Makerere AI Lab Beans dataset and
established open-source machine learning and software
engineering libraries.

The MobileNetV2 experiment incorporates pretrained ImageNet
weights through TensorFlow/Keras. The pretrained backbone
is an external resource rather than a model trained from
scratch by the project team.

AI-assisted tools, including OpenAI ChatGPT, were used during
project development to support code validation, debugging
and implementation refinement. The project implementation
was tested and verified by the contributors, and all reported
experimental results are based on actual executions and
recorded outputs.

### Technical references

- Beans dataset:
  https://huggingface.co/datasets/AI-Lab-Makerere/beans
- TensorFlow:
  https://www.tensorflow.org/
- Keras:
  https://keras.io/
- MLflow:
  https://mlflow.org/
- FastAPI:
  https://fastapi.tiangolo.com/
- Docker:
  https://docs.docker.com/
- Pytest:
  https://docs.pytest.org/
- GitHub Actions:
  https://docs.github.com/en/actions

- MobileNetV2 architecture:

  https://arxiv.org/abs/1801.04381

- Keras MobileNet models and pretrained weights:

  https://keras.io/api/applications/mobilenet/

- OpenAI ChatGPT:

  https://chatgpt.com/
