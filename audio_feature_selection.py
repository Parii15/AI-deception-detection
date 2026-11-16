
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
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
OUT = ROOT / "results/audio_feature_selection_v1"

OUT.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(DATA / "train_audio_facial.csv")
manifest = pd.read_csv(MANIFEST)

source_map = (
    manifest[["clip", "youtube_id"]]
    .drop_duplicates("clip")
)

data = data.merge(
    source_map,
    on="clip",
    how="left",
    validate="one_to_one"
)

if data["youtube_id"].isna().any():
    raise ValueError("Missing YouTube source IDs.")

# Audio features only
audio_cols = [
    c for c in data.columns
    if c.startswith("audio_") or c.startswith("mfcc_")
]

X = data[audio_cols].apply(pd.to_numeric, errors="coerce")
y = data["label"]
groups = data["youtube_id"]

print("Total audio features:", len(audio_cols))
print("Training clips:", len(data))
print("YouTube source groups:", groups.nunique())

# Feature selection is INSIDE the pipeline.
# Therefore it is fitted only on each training fold.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif)),
    ("scaler", StandardScaler()),
    ("model", SVC(class_weight="balanced"))
])

param_grid = [
    {
        "selector__k": [10, 20, 30, 40, "all"],
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10]
    },
    {
        "selector__k": [10, 20, 30, 40, "all"],
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.001, 0.01]
    }
]

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = pd.Series(index=data.index, dtype="object")
fold_results = []
selected_features = []

for fold, (train_idx, val_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):
    print(f"\nOuter fold {fold}/5")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    train_groups = groups.iloc[train_idx]

    inner_cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=100 + fold
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

    best_model = search.best_estimator_
    pred = best_model.predict(X_val)

    predictions.iloc[val_idx] = pred

    selector = best_model.named_steps["selector"]
    chosen = [
        audio_cols[i]
        for i in selector.get_support(indices=True)
    ]

    selected_features.append({
        "Fold": fold,
        "Selected Feature Count": len(chosen),
        "Features": ", ".join(chosen)
    })

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy_score(y_val, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_val, pred),
        "Precision": precision_score(
            y_val, pred,
            pos_label="deception",
            zero_division=0
        ),
        "Recall": recall_score(
            y_val, pred,
            pos_label="deception",
            zero_division=0
        ),
        "F1": f1_score(
            y_val, pred,
            pos_label="deception",
            zero_division=0
        ),
        "Best Parameters": str(search.best_params_),
        "Inner CV Balanced Accuracy": search.best_score_
    })

    print("Best parameters:", search.best_params_)
    print("Selected features:", len(chosen))
    print("Outer accuracy:", round(accuracy_score(y_val, pred), 4))

if predictions.isna().any():
    raise ValueError("Some clips have missing predictions.")

overall = {
    "Accuracy": accuracy_score(y, predictions),
    "Balanced Accuracy": balanced_accuracy_score(y, predictions),
    "Precision": precision_score(
        y, predictions,
        pos_label="deception",
        zero_division=0
    ),
    "Recall": recall_score(
        y, predictions,
        pos_label="deception",
        zero_division=0
    ),
    "F1": f1_score(
        y, predictions,
        pos_label="deception",
        zero_division=0
    )
}

fold_df = pd.DataFrame(fold_results)
features_df = pd.DataFrame(selected_features)

fold_df.to_csv(OUT / "fold_results.csv", index=False)
features_df.to_csv(OUT / "selected_features_by_fold.csv", index=False)

pd.DataFrame([overall]).to_csv(
    OUT / "overall_results.csv",
    index=False
)

print("\n===== AUDIO FEATURE SELECTION RESULTS =====")
print(pd.DataFrame([overall]).to_string(index=False))

print("\n===== FOLD RESULTS =====")
print(fold_df.to_string(index=False))

print("\n===== SELECTED FEATURES BY FOLD =====")
print(features_df[["Fold", "Selected Feature Count"]].to_string(index=False))

print("\nAll results saved in:", OUT)
print("AUDIO FEATURE SELECTION COMPLETED")
