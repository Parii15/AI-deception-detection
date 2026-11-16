
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data/DOLOS/bulk_processed/features_face_fusion_v2"

MODEL_PATH = (
    ROOT / "models/facial_fusion_v1/"
    "dolos_audio_facial_best.joblib"
)

RESULTS = ROOT / "results/facial_fusion_v1"

train = pd.read_csv(DATA / "train_audio_facial.csv")
test = pd.read_csv(DATA / "test_audio_facial.csv")

package = joblib.load(MODEL_PATH)
fusion_model = package["model"]

y_train = train["label"]
y_test = test["label"]

# Use only audio + MFCC columns
audio_cols = [
    c for c in train.columns
    if c.startswith("audio_") or c.startswith("mfcc_")
]

X_train = train[audio_cols].apply(
    pd.to_numeric, errors="coerce"
)

X_test = test[audio_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Match the selected SVM's hyperparameters
params = fusion_model.named_steps["model"].get_params()

audio_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        class_weight="balanced"
    ))
])

audio_model.fit(X_train, y_train)

audio_predictions = audio_model.predict(X_test)
fusion_predictions = fusion_model.predict(
    test[package["feature_columns"]]
)

def metrics(name, actual, predicted):

    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced Accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "Precision": precision_score(
            actual, predicted,
            pos_label="deception",
            zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted,
            pos_label="deception",
            zero_division=0
        ),
        "F1": f1_score(
            actual, predicted,
            pos_label="deception",
            zero_division=0
        )
    }

comparison = pd.DataFrame([
    metrics("Audio only", y_test, audio_predictions),
    metrics("Audio + Facial", y_test, fusion_predictions)
])

print("\n===== FAIR SAME-CLIP COMPARISON =====")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS / "audio_vs_fusion_same_clips.csv",
    index=False
)

print("\nSaved comparison:", RESULTS / "audio_vs_fusion_same_clips.csv")
