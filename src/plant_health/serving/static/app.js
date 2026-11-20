"use strict";

const MAX_FILE_BYTES = 8 * 1024 * 1024;
const SUPPORTED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const previewContainer = document.getElementById("preview-container");
const fileName = document.getElementById("file-name");
const analyzeButton = document.getElementById("analyze-button");
const loadingMessage = document.getElementById("loading-message");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results-section");

let selectedFile = null;
let previewUrl = null;
let analyzing = false;

function formatClassName(value) {
  return String(value).replaceAll("_", " ").replaceAll("-", " ");
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function clearPreview() {
  if (previewUrl !== null) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  imagePreview.removeAttribute("src");
  fileName.textContent = "";
  previewContainer.hidden = true;
}

function setLoading(isLoading) {
  analyzing = isLoading;
  analyzeButton.disabled = isLoading || selectedFile === null;
  analyzeButton.textContent = isLoading
    ? "Analyzing..."
    : "Analyze leaf";
  loadingMessage.hidden = !isLoading;
}

imageInput.addEventListener("change", () => {
  if (analyzing) {
    return;
  }

  clearError();
  resultsSection.hidden = true;
  selectedFile = null;
  clearPreview();

  const file = imageInput.files?.[0];

  if (!file) {
    analyzeButton.disabled = true;
    return;
  }

  if (!SUPPORTED_TYPES.has(file.type)) {
    showError("Please select a JPEG, PNG, or WebP image.");
    imageInput.value = "";
    analyzeButton.disabled = true;
    return;
  }

  if (file.size > MAX_FILE_BYTES) {
    showError("The image exceeds the 8 MB upload limit.");
    imageInput.value = "";
    analyzeButton.disabled = true;
    return;
  }

  if (file.size === 0) {
    showError("The selected image is empty.");
    imageInput.value = "";
    analyzeButton.disabled = true;
    return;
  }

  selectedFile = file;
  previewUrl = URL.createObjectURL(file);

  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  previewContainer.hidden = false;
  analyzeButton.disabled = false;
});

function renderProbabilities(probabilities) {
  const container = document.getElementById("probabilities-list");
  container.replaceChildren();

  const entries = Object.entries(probabilities || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  for (const [className, probability] of entries) {
    const item = document.createElement("div");
    item.className = "probability-item";

    const header = document.createElement("div");
    header.className = "probability-header";

    const label = document.createElement("span");
    label.textContent = formatClassName(className);

    const value = document.createElement("strong");
    value.textContent = formatPercent(probability);

    header.append(label, value);

    const track = document.createElement("div");
    track.className = "progress-track";

    const bar = document.createElement("div");
    bar.className = "progress-bar";
    bar.style.width = `${Math.max(
      0,
      Math.min(100, Number(probability) * 100)
    )}%`;

    track.appendChild(bar);
    item.append(header, track);
    container.appendChild(item);
  }
}

function renderResults(data) {
  document.getElementById("predicted-class").textContent =
    formatClassName(data.predicted_class);

  document.getElementById("confidence-value").textContent =
    formatPercent(data.confidence);

  const confidence = Math.max(
    0,
    Math.min(100, Number(data.confidence) * 100)
  );

  document.getElementById("confidence-bar").style.width =
    `${confidence}%`;

  document.getElementById("confidence-track").setAttribute(
    "aria-valuenow",
    String(Math.round(confidence))
  );

  const statusBadge = document.getElementById("prediction-status");
  statusBadge.classList.remove("warning");

  const qualityMessage = document.getElementById("quality-message");

  if (data.status === "low_quality") {
    statusBadge.textContent = "Low image quality";
    statusBadge.classList.add("warning");
    qualityMessage.textContent =
      "The image may be too dark, too bright, or blurry. " +
      "Try taking another photo with better lighting and focus.";
  } else if (data.status === "uncertain") {
    statusBadge.textContent = "Uncertain prediction";
    statusBadge.classList.add("warning");
    qualityMessage.textContent =
      "The model has low confidence in this prediction. " +
      "Try a clearer photo of a single leaf.";
  } else {
    statusBadge.textContent = "Analysis complete";
    qualityMessage.textContent =
      "No image-quality warning was triggered. " +
      "This does not guarantee prediction accuracy.";
  }

  renderProbabilities(data.probabilities);

  document.getElementById("model-version").textContent =
    `Model version: ${data.model_version || "unknown"}`;

  resultsSection.hidden = false;
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || analyzing) {
    return;
  }

  clearError();
  resultsSection.hidden = true;
  setLoading(true);
  imageInput.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `Prediction failed (HTTP ${response.status}).`;

      try {
        const errorData = await response.json();
        if (typeof errorData.detail === "string") {
          message = errorData.detail;
        }
      } catch {
        // Keep the generic HTTP error message.
      }

      throw new Error(message);
    }

    const data = await response.json();
    renderResults(data);
    resultsSection.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  } catch (error) {
    showError(
      error instanceof Error
        ? error.message
        : "An unexpected error occurred."
    );
  } finally {
    imageInput.disabled = false;
    setLoading(false);
  }
});

window.addEventListener("pagehide", () => {
  if (previewUrl !== null) {
    URL.revokeObjectURL(previewUrl);
  }
});
