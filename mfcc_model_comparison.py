
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
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
OUT = ROOT / "results/mfcc_model_comparison_v1"
OUT.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA / "train_audio_facial.csv")
manifest = pd.read_csv(MANIFEST)

source_map = manifest[["clip", "youtube_id"]].drop_duplicates("clip")
data = data.merge(source_map, on="clip", how="left", validate="one_to_one")

if data["youtube_id"].isna().any():
    raise ValueError("Missing YouTube source IDs")

mfcc_cols = [c for c in data.columns if c.startswith("mfcc_")]

X = data[mfcc_cols].apply(pd.to_numeric, errors="coerce")
y = data["label"]
groups = data["youtube_id"]

models = {
    "SVM": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced"))
        ]),
        [
            {"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
            {
                "model__kernel": ["rbf"],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.001, 0.01]
            }
        ]
    ),

    "Logistic Regression": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=3000
            ))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Extra Trees": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            ))
        ]),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", 0.7]
        }
    )
}

outer_cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

all_results = []

for name, (pipeline, param_grid) in models.items():
    print(f"\n===== TESTING {name} =====")

    predictions = pd.Series(index=data.index, dtype="object")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
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

        fold_results.append({
            "Model": name,
            "Fold": fold,
            "Accuracy": accuracy_score(y_val, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_val, pred),
            "F1": f1_score(
                y_val, pred,
                pos_label="deception",
                zero_division=0
            ),
            "Best Parameters": str(search.best_params_)
        })

        print(
            f"Fold {fold}: accuracy = "
            f"{accuracy_score(y_val, pred):.4f}"
        )

    if predictions.isna().any():
        raise ValueError(f"Missing predictions for {name}")

    all_results.append({
        "Model": name,
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
    })

    pd.DataFrame(fold_results).to_csv(
        OUT / f"{name.lower().replace(' ', '_')}_folds.csv",
        index=False
    )

results = pd.DataFrame(all_results)

print("\n===== MFCC MODEL COMPARISON =====")
print(results.to_string(index=False))

results.to_csv(OUT / "model_comparison.csv", index=False)

print("\nSaved safely in:", OUT)
print("MFCC MODEL COMPARISON COMPLETED")
