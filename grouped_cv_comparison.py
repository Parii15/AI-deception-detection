
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
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
MANIFEST = ROOT / "data/DOLOS/bulk_processed/manifests/train.csv"

RESULTS = ROOT / "results/facial_fusion_v1"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load training features and source information
train = pd.read_csv(DATA / "train_audio_facial.csv")
manifest = pd.read_csv(MANIFEST)

# One source ID per clip
source_map = manifest[["clip", "youtube_id"]].drop_duplicates("clip")

data = train.merge(
    source_map,
    on="clip",
    how="left",
    validate="one_to_one"
)

if data["youtube_id"].isna().any():
    raise ValueError("Some clips have no youtube_id. Check the manifest.")

# Select features
audio_cols = [
    c for c in data.columns
    if c.startswith("audio_") or c.startswith("mfcc_")
]

facial_cols = [
    "face_detection_rate",
    "eye_detection_rate",
    "eye_count_mean",
    "smile_detection_rate",
    "face_size_mean",
    "face_size_std",
    "head_center_motion",
    "mouth_region_change"
]

fusion_cols = audio_cols + facial_cols

missing = [c for c in fusion_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

y = data["label"]
groups = data["youtube_id"]

X_audio = data[audio_cols].apply(pd.to_numeric, errors="coerce")
X_fusion = data[fusion_cols].apply(pd.to_numeric, errors="coerce")

# Identical grouped folds for both models
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            C=0.01,
            class_weight="balanced"
        ))
    ])

predictions = {
    "Audio only": pd.Series(index=data.index, dtype="object"),
    "Audio + Facial": pd.Series(index=data.index, dtype="object")
}

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_fusion, y, groups),
    start=1
):
    print(f"\nRunning fold {fold}/5...")

    for name, X in [
        ("Audio only", X_audio),
        ("Audio + Facial", X_fusion)
    ]:
        model = make_model()

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])

        predictions[name].iloc[val_idx] = pred

        fold_results.append({
            "Fold": fold,
            "Model": name,
            "Accuracy": accuracy_score(y.iloc[val_idx], pred),
            "Balanced Accuracy": balanced_accuracy_score(
                y.iloc[val_idx], pred
            ),
            "Precision": precision_score(
                y.iloc[val_idx], pred,
                pos_label="deception", zero_division=0
            ),
            "Recall": recall_score(
                y.iloc[val_idx], pred,
                pos_label="deception", zero_division=0
            ),
            "F1": f1_score(
                y.iloc[val_idx], pred,
                pos_label="deception", zero_division=0
            )
        })

# Overall out-of-fold metrics
overall = []

for name, pred in predictions.items():
    if pred.isna().any():
        raise ValueError(f"Missing predictions for {name}")

    overall.append({
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(
            y, pred, pos_label="deception", zero_division=0
        ),
        "Recall": recall_score(
            y, pred, pos_label="deception", zero_division=0
        ),
        "F1": f1_score(
            y, pred, pos_label="deception", zero_division=0
        )
    })

overall_df = pd.DataFrame(overall)
fold_df = pd.DataFrame(fold_results)

print("\n===== GROUPED 5-FOLD OVERALL RESULTS =====")
print(overall_df.to_string(index=False))

print("\n===== RESULTS BY FOLD =====")
print(fold_df.to_string(index=False))

overall_df.to_csv(
    RESULTS / "grouped_cv_overall.csv", index=False
)

fold_df.to_csv(
    RESULTS / "grouped_cv_by_fold.csv", index=False
)

print("\nResults saved in:", RESULTS)
print("GROUPED CROSS-VALIDATION COMPLETED")
