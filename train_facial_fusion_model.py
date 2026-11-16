
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data/DOLOS/bulk_processed/features_face_fusion_v2"
MANIFEST_DIR = ROOT / "data/DOLOS/bulk_processed/manifests"

MODEL_DIR = ROOT / "models/facial_fusion_v1"
RESULT_DIR = ROOT / "results/facial_fusion_v1"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(split):
    path = DATA_DIR / f"{split}_audio_facial.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if df["clip"].duplicated().any():
        raise ValueError(f"Duplicate clips found in {split} data")

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    allowed = {"truth", "deception"}
    if not set(df["label"].unique()).issubset(allowed):
        raise ValueError(f"Unexpected labels in {split}")

    return df


def load_groups(split, clips):
    manifest_path = MANIFEST_DIR / f"{split}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    required = {"clip", "youtube_id"}
    if not required.issubset(manifest.columns):
        raise ValueError("Manifest must contain clip and youtube_id columns")

    manifest = manifest.drop_duplicates("clip")

    group_map = manifest.set_index("clip")["youtube_id"]

    groups = clips.map(group_map)

    if groups.isna().any():
        missing = clips[groups.isna()].tolist()[:10]
        raise ValueError(f"Missing source IDs for clips: {missing}")

    return groups.astype(str).to_numpy()


def make_pipeline(model, scale=True):
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps)


def main():

    print("\n====================================")
    print(" AUDIO + FACIAL FUSION MODEL")
    print("====================================")

    train = load_data("train")
    test = load_data("test")

    feature_cols = [
        col for col in train.columns
        if col not in ["clip", "label"]
    ]

    if set(feature_cols) != {
        col for col in test.columns
        if col not in ["clip", "label"]
    }:
        raise ValueError("Train and test feature columns do not match")

    X_train = train[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    groups = load_groups("train", train["clip"])

    print("\nTRAIN CLIPS:", len(train))
    print("TEST CLIPS:", len(test))
    print("FEATURES:", len(feature_cols))
    print("TRAIN SOURCE GROUPS:", len(np.unique(groups)))

    print("\nTrain labels:")
    print(train["label"].value_counts())

    print("\nTest labels:")
    print(test["label"].value_counts())

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_splits = list(cv.split(X_train, y_train, groups))

    models = {
        "SVM": {
            "pipeline": make_pipeline(
                SVC(class_weight="balanced"),
                scale=True
            ),
            "params": {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        },

        "Random Forest": {
            "pipeline": make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ),
                scale=False
            ),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 15, 25],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7]
            }
        },

        "Extra Trees": {
            "pipeline": make_pipeline(
                ExtraTreesClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ),
                scale=False
            ),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 15, 25],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7]
            }
        },

        "Logistic Regression": {
            "pipeline": make_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42
                ),
                scale=True
            ),
            "params": {
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__solver": ["liblinear"]
            }
        },

        "Gradient Boosting": {
            "pipeline": make_pipeline(
                HistGradientBoostingClassifier(
                    random_state=42
                ),
                scale=False
            ),
            "params": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 200, 300],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0, 0.1, 1]
            }
        }
    }

    results = []
    best_search = None
    best_name = None
    best_cv_score = -1

    print("\nStarting grouped cross-validation...")

    for name, config in models.items():

        print("\n------------------------------------")
        print("MODEL:", name)
        print("------------------------------------")

        search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["params"],
            n_iter=12,
            scoring="balanced_accuracy",
            cv=cv_splits,
            random_state=42,
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        score = search.best_score_

        print("CV balanced accuracy:", round(score, 4))
        print("Best parameters:", search.best_params_)

        results.append({
            "model": name,
            "cv_balanced_accuracy": score,
            "best_parameters": str(search.best_params_)
        })

        if score > best_cv_score:
            best_cv_score = score
            best_search = search
            best_name = name

    comparison = pd.DataFrame(results).sort_values(
        "cv_balanced_accuracy",
        ascending=False
    )

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n====================================")
    print("CROSS-VALIDATION COMPARISON")
    print("====================================")
    print(comparison.to_string(index=False))

    # Evaluate the selected model only once on held-out test data.
    best_model = best_search.best_estimator_
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        pos_label="deception",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        pos_label="deception",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        pos_label="deception",
        zero_division=0
    )

    print("\n====================================")
    print("FINAL FUSION TEST RESULTS")
    print("====================================")
    print("Selected model:", best_name)
    print("CV balanced accuracy:", round(best_cv_score, 4))
    print("Test accuracy:", round(accuracy, 4))
    print("Test balanced accuracy:", round(balanced, 4))
    print("Test precision:", round(precision, 4))
    print("Test recall:", round(recall, 4))
    print("Test F1:", round(f1, 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["truth", "deception"],
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["truth", "deception"]
    )

    print("\nConfusion matrix (Truth, Deception):")
    print(cm)

    # Save the model separately from all previous models.
    model_package = {
        "model": best_model,
        "feature_columns": feature_cols,
        "selected_model": best_name,
        "cv_balanced_accuracy": float(best_cv_score),
        "labels": ["truth", "deception"]
    }

    model_path = MODEL_DIR / "dolos_audio_facial_best.joblib"
    joblib.dump(model_package, model_path)

    metrics = {
        "selected_model": best_name,
        "cv_balanced_accuracy": float(best_cv_score),
        "test_accuracy": float(accuracy),
        "test_balanced_accuracy": float(balanced),
        "test_precision_deception": float(precision),
        "test_recall_deception": float(recall),
        "test_f1_deception": float(f1),
        "test_clips": int(len(test)),
        "train_clips": int(len(train)),
        "feature_count": int(len(feature_cols)),
        "confusion_matrix_labels": ["truth", "deception"],
        "confusion_matrix": cm.tolist()
    }

    with open(
        RESULT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    prediction_output = pd.DataFrame({
        "clip": test["clip"],
        "actual_label": y_test,
        "predicted_label": predictions
    })

    prediction_output.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False
    )

    print("\n====================================")
    print("FUSION TRAINING COMPLETED")
    print("====================================")
    print("Model saved:", model_path)
    print("Comparison saved:", RESULT_DIR / "model_comparison.csv")
    print("Metrics saved:", RESULT_DIR / "metrics.json")
    print("Predictions saved:", RESULT_DIR / "test_predictions.csv")


if __name__ == "__main__":
    main()
