
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
OUT = ROOT / "results/audio_tuning_v1"
OUT.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA / "train_audio_facial.csv")
manifest = pd.read_csv(MANIFEST)

source_map = manifest[["clip", "youtube_id"]].drop_duplicates("clip")
data = data.merge(source_map, on="clip", how="left", validate="one_to_one")

if data["youtube_id"].isna().any():
    raise ValueError("Missing YouTube source IDs.")

audio_cols = [
    c for c in data.columns
    if c.startswith("audio_") or c.startswith("mfcc_")
]

X = data[audio_cols].apply(pd.to_numeric, errors="coerce")
y = data["label"]
groups = data["youtube_id"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(class_weight="balanced"))
])

param_grid = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1]
    }
]

outer_cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

predictions = pd.Series(index=data.index, dtype="object")
fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    outer_cv.split(X, y, groups), start=1
):
    print(f"\nOuter fold {fold}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    train_groups = groups.iloc[train_idx]

    inner_cv = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=100 + fold
    )

    inner_splits = list(
        inner_cv.split(X_train, y_train, train_groups)
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=inner_splits,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)
    pred = search.predict(X_val)
    predictions.iloc[val_idx] = pred

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy_score(y_val, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_val, pred),
        "Precision": precision_score(
            y_val, pred, pos_label="deception", zero_division=0
        ),
        "Recall": recall_score(
            y_val, pred, pos_label="deception", zero_division=0
        ),
        "F1": f1_score(
            y_val, pred, pos_label="deception", zero_division=0
        ),
        "Best Parameters": str(search.best_params_),
        "Inner CV Balanced Accuracy": search.best_score_
    })

    print("Best parameters:", search.best_params_)
    print("Outer fold accuracy:", round(accuracy_score(y_val, pred), 4))

if predictions.isna().any():
    raise ValueError("Some clips have no out-of-fold predictions.")

overall = {
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

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(OUT / "nested_cv_folds.csv", index=False)

pd.DataFrame([overall]).to_csv(
    OUT / "nested_cv_overall.csv", index=False
)

print("\n===== NESTED GROUPED CV RESULTS =====")
print(pd.DataFrame([overall]).to_string(index=False))

print("\n===== FOLD DETAILS =====")
print(fold_df.to_string(index=False))

print("\nResults saved:", OUT)
print("AUDIO MODEL TUNING COMPLETED")
