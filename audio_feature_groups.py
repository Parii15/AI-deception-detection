
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
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
OUT = ROOT / "results/audio_feature_groups_v1"
OUT.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA / "train_audio_facial.csv")
manifest = pd.read_csv(MANIFEST)

source_map = manifest[["clip", "youtube_id"]].drop_duplicates("clip")
data = data.merge(source_map, on="clip", how="left", validate="one_to_one")

if data["youtube_id"].isna().any():
    raise ValueError("Missing YouTube source IDs")

# Feature groups
audio_cols = [
    c for c in data.columns
    if c.startswith("audio_")
]

mfcc_cols = [
    c for c in data.columns
    if c.startswith("mfcc_")
]

groups_to_test = {
    "Audio basic": audio_cols,
    "MFCC only": mfcc_cols,
    "All audio": audio_cols + mfcc_cols
}

y = data["label"]
source_groups = data["youtube_id"]

outer_cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

param_grid = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10]
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.001, 0.01]
    }
]

all_results = []

for group_name, columns in groups_to_test.items():

    if not columns:
        print(f"Skipping {group_name}: no features found")
        continue

    print(f"\n===== TESTING {group_name} ({len(columns)} features) =====")

    X = data[columns].apply(pd.to_numeric, errors="coerce")

    predictions = pd.Series(index=data.index, dtype="object")

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y, source_groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        train_groups = source_groups.iloc[train_idx]

        inner_cv = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=100 + fold
        )

        inner_splits = list(
            inner_cv.split(X_train, y_train, train_groups)
        )

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced"))
        ])

        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="balanced_accuracy",
            cv=inner_splits,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)
        pred = search.predict(X_val)

        predictions.iloc[val_idx] = pred

        print(
            f"Fold {fold}: accuracy = "
            f"{accuracy_score(y_val, pred):.4f}"
        )

    if predictions.isna().any():
        raise ValueError(f"Missing predictions: {group_name}")

    result = {
        "Feature Group": group_name,
        "Feature Count": len(columns),
        "Accuracy": accuracy_score(y, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, pos_label="deception", zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, pos_label="deception", zero_division=0
        ),
        "F1": f1_score(
            y, predictions, pos_label="deception", zero_division=0
        )
    }

    all_results.append(result)

results = pd.DataFrame(all_results)

print("\n===== FEATURE GROUP COMPARISON =====")
print(results.to_string(index=False))

results.to_csv(
    OUT / "feature_group_comparison.csv", index=False
)

print("\nSaved safely in:", OUT)
print("FEATURE GROUP EXPERIMENT COMPLETED")
