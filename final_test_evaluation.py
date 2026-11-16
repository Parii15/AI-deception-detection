import os
import json
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE = os.getcwd()

TRAIN_PATH = os.path.join(
    BASE, "data", "DOLOS", "bulk_processed",
    "features_face_fusion_v2", "train_audio_facial.csv"
)

TEST_PATH = os.path.join(
    BASE, "data", "DOLOS", "bulk_processed",
    "features_face_fusion_v2", "test_audio_facial.csv"
)

OUTPUT_DIR = os.path.join(
    BASE, "results", "final_test_mfcc_v1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

feature_cols = [
    c for c in train.columns if c.startswith("mfcc_")
]

X_train = train[feature_cols]
y_train = train["label"]

X_test = test[feature_cols]
y_test = test["label"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1,
        gamma=0.001,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, pos_label="deception", zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, pos_label="deception", zero_division=0
    ),
    "f1": f1_score(
        y_test, predictions, pos_label="deception", zero_division=0
    )
}

print("\n===== FINAL TEST RESULTS =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "clip": test["clip"],
    "actual": y_test,
    "predicted": predictions
}).to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)

joblib.dump(
    model,
    os.path.join(OUTPUT_DIR, "final_mfcc_svm.joblib")
)

print("\nSaved safely in:", OUTPUT_DIR)