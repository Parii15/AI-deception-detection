
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parent

FEATURE_DIR = ROOT / "data" / "DOLOS" / "bulk_processed" / "features"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load extracted features
train_df = pd.read_csv(FEATURE_DIR / "train_features.csv")
test_df = pd.read_csv(FEATURE_DIR / "test_features.csv")

# Identify audio and visual features
feature_cols = [
    c for c in train_df.columns
    if c.startswith("audio_") or c.startswith("visual_")
]

if len(feature_cols) != 12:
    raise ValueError(f"Expected 12 features, found {len(feature_cols)}")

if feature_cols != [
    c for c in test_df.columns
    if c.startswith("audio_") or c.startswith("visual_")
]:
    raise ValueError("Train and test feature columns do not match")

# Prepare inputs and labels
X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

y_train = train_df["label"].str.strip().str.lower()
y_test = test_df["label"].str.strip().str.lower()

# Check labels
allowed_labels = {"truth", "deception"}

if set(y_train.unique()) != allowed_labels:
    raise ValueError(f"Unexpected training labels: {y_train.unique()}")

if set(y_test.unique()) != allowed_labels:
    raise ValueError(f"Unexpected testing labels: {y_test.unique()}")

# Check for overlapping clips
overlap = set(train_df["clip"]) & set(test_df["clip"])

if overlap:
    raise ValueError(f"Train/test clip overlap: {len(overlap)} clips")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", len(feature_cols))
print("\nTraining model...")

# Random Forest classifier
model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, pos_label="deception"
)
recall = recall_score(
    y_test, predictions, pos_label="deception"
)
f1 = f1_score(
    y_test, predictions, pos_label="deception"
)
balanced_acc = balanced_accuracy_score(y_test, predictions)

print("\n===== AUDIO-VISUAL MODEL RESULTS =====")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

print("\nClassification Report:")
report = classification_report(y_test, predictions)
print(report)

# Save trained model and feature order
model_path = MODEL_DIR / "dolos_av_random_forest.joblib"

joblib.dump({
    "model": model,
    "feature_columns": feature_cols
}, model_path)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "balanced_accuracy": float(balanced_acc),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "features": feature_cols
}

with open(RESULT_DIR / "av_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["truth", "deception"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Truth", "Deception"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Audio-Visual Deception Detection")
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "av_confusion_matrix.png",
    dpi=150
)
plt.close()

print("\nModel saved:", model_path)
print("Metrics saved:", RESULT_DIR / "av_model_metrics.json")
print("Confusion matrix saved:", RESULT_DIR / "av_confusion_matrix.png")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
