
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "data" / "DOLOS" / "bulk_processed" / "features"
MANIFEST_DIR = ROOT / "data" / "DOLOS" / "bulk_processed" / "manifests"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Load data
train = pd.read_csv(FEATURE_DIR / "train_features.csv")
test = pd.read_csv(FEATURE_DIR / "test_features.csv")
manifest = pd.read_csv(MANIFEST_DIR / "train.csv")

# Remove duplicate training clips safely
if train["clip"].duplicated().any():
    conflicts = train.groupby("clip")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Duplicate clips have conflicting labels")

    train = train.drop_duplicates(
        subset="clip", keep="first"
    ).reset_index(drop=True)

# Remove duplicate manifest entries safely
if manifest["clip"].duplicated().any():
    conflicts = manifest.groupby("clip")["youtube_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Duplicate clips have conflicting source IDs")

    manifest = manifest.drop_duplicates(
        subset="clip", keep="first"
    ).reset_index(drop=True)

features = [
    c for c in train.columns
    if c.startswith("audio_") or c.startswith("visual_")
]

# Keep only numerical audio-visual features
X_train = train[features].apply(pd.to_numeric, errors="coerce")
X_test = test[features].apply(pd.to_numeric, errors="coerce")

y_train = train["label"].str.strip().str.lower()
y_test = test["label"].str.strip().str.lower()

# Group clips by original YouTube source
group_map = manifest.set_index("clip")["youtube_id"]
groups = train["clip"].map(group_map)

if groups.isna().any():
    raise ValueError("Some training clips have no youtube_id in train.csv")

if len(features) != 12:
    raise ValueError(f"Expected 12 features, found {len(features)}")

if set(y_train.unique()) != {"truth", "deception"}:
    raise ValueError("Unexpected training labels")

if set(y_test.unique()) != {"truth", "deception"}:
    raise ValueError("Unexpected test labels")

# Check source overlap
test_manifest = pd.read_csv(MANIFEST_DIR / "test.csv")
train_sources = set(groups)
test_sources = set(test_manifest["youtube_id"])

if train_sources.intersection(test_sources):
    raise ValueError("YouTube source leakage detected between train and test")

# Group-aware cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Candidate models and hyperparameters
searches = {
    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        ),
        {
            "model__n_estimators": [300, 600],
            "model__max_features": ["sqrt", 0.8],
            "model__min_samples_leaf": [1, 2, 4]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        ),
        {
            "model__n_estimators": [300, 600],
            "model__max_features": ["sqrt", 0.8],
            "model__min_samples_leaf": [1, 2, 4]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2, 3]
        }
    ),

    "SVM": (
        SVC(class_weight="balanced"),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", 0.01]
        }
    )
}

leaderboard = []
best_search = None
best_score = -1
best_name = None

for name, (estimator, params) in searches.items():
    print(f"\nTuning {name}...")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    search.fit(X_train, y_train, groups=groups)

    score = search.best_score_

    leaderboard.append({
        "model": name,
        "cv_balanced_accuracy": score,
        "best_parameters": str(search.best_params_)
    })

    print(f"{name} CV balanced accuracy: {score:.4f}")
    print("Best parameters:", search.best_params_)

    if score > best_score:
        best_score = score
        best_search = search
        best_name = name

# Save CV comparison
leaderboard_df = pd.DataFrame(leaderboard)
leaderboard_df = leaderboard_df.sort_values(
    "cv_balanced_accuracy", ascending=False
)
leaderboard_df.to_csv(
    RESULT_DIR / "av_model_comparison.csv", index=False
)

print("\n===== CROSS-VALIDATION LEADERBOARD =====")
print(leaderboard_df.to_string(index=False))

# Evaluate the selected model on untouched test data ONCE
best_model = best_search.best_estimator_
predictions = best_model.predict(X_test)

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

print("\n===== FINAL TEST RESULTS =====")
print("Selected model:", best_name)
print(f"Cross-validation balanced accuracy: {best_score:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"Test balanced accuracy: {balanced_acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save final model
joblib.dump({
    "model": best_model,
    "feature_columns": features,
    "model_name": best_name
}, MODEL_DIR / "dolos_av_best_model.joblib")

# Save final metrics
metrics = {
    "selected_model": best_name,
    "cv_balanced_accuracy": float(best_score),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_balanced_accuracy": float(balanced_acc)
}

with open(RESULT_DIR / "av_best_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix
cm = confusion_matrix(
    y_test, predictions,
    labels=["truth", "deception"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Truth", "Deception"]
)
display.plot(cmap="Blues", values_format="d")
plt.title(f"Best Audio-Visual Model: {best_name}")
plt.tight_layout()
plt.savefig(RESULT_DIR / "av_best_confusion_matrix.png", dpi=150)
plt.close()

print("\nBest model saved successfully!")
print("MODEL:", MODEL_DIR / "dolos_av_best_model.joblib")
print("COMPARISON:", RESULT_DIR / "av_model_comparison.csv")
print("METRICS:", RESULT_DIR / "av_best_model_metrics.json")
