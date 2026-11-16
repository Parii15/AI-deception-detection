
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "DOLOS" / "bulk_processed"
FEATURES = DATA / "features_v2"
MANIFESTS = DATA / "manifests"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Load V2 data
train = pd.read_csv(FEATURES / "train_features.csv")
test = pd.read_csv(FEATURES / "test_features.csv")
manifest = pd.read_csv(MANIFESTS / "train.csv")
test_manifest = pd.read_csv(MANIFESTS / "test.csv")

# Remove duplicate training clips consistently
train = train.drop_duplicates("clip", keep="first").reset_index(drop=True)
manifest = manifest.drop_duplicates("clip", keep="first").reset_index(drop=True)

# Use ALL numerical features except clip and label
feature_cols = [
    c for c in train.columns
    if c not in ["clip", "label"]
]

X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")

y_train = train["label"].str.strip().str.lower()
y_test = test["label"].str.strip().str.lower()

if len(feature_cols) != 54:
    raise ValueError(f"Expected 54 features, found {len(feature_cols)}")

if feature_cols != [
    c for c in test.columns if c not in ["clip", "label"]
]:
    raise ValueError("Train/test feature columns do not match")

# Map each clip to its original YouTube source
group_map = manifest.set_index("clip")["youtube_id"]
groups = train["clip"].map(group_map)

if groups.isna().any():
    raise ValueError("Some clips have no YouTube source ID")

# Verify no source leakage
train_sources = set(groups)
test_sources = set(test_manifest["youtube_id"])

if train_sources.intersection(test_sources):
    raise ValueError("Train/test YouTube source overlap detected")

print("Training clips:", len(train))
print("Testing clips:", len(test))
print("Features:", len(feature_cols))
print("Unique training sources:", groups.nunique())

# Group-aware cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Models to compare
searches = {
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

    "SVM": (
        SVC(class_weight="balanced"),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }
    )
}

leaderboard = []
best_search = None
best_score = -1
best_name = None

for name, (estimator, params) in searches.items():
    print(f"\nTraining and tuning: {name}")

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

    print(f"CV balanced accuracy: {score:.4f}")
    print("Best parameters:", search.best_params_)

    if score > best_score:
        best_score = score
        best_search = search
        best_name = name

# Save comparison
leaderboard_df = pd.DataFrame(leaderboard)
leaderboard_df = leaderboard_df.sort_values(
    "cv_balanced_accuracy", ascending=False
)
leaderboard_df.to_csv(
    RESULTS / "av_v2_model_comparison.csv",
    index=False
)

print("\n===== V2 CROSS-VALIDATION RESULTS =====")
print(leaderboard_df.to_string(index=False))

# Evaluate best model on untouched test data
best_model = best_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label="deception")
recall = recall_score(y_test, predictions, pos_label="deception")
f1 = f1_score(y_test, predictions, pos_label="deception")
balanced_acc = balanced_accuracy_score(y_test, predictions)

print("\n===== V2 FINAL TEST RESULTS =====")
print("Selected model:", best_name)
print(f"CV balanced accuracy: {best_score:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"Test balanced accuracy: {balanced_acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump({
    "model": best_model,
    "feature_columns": feature_cols,
    "model_name": best_name
}, MODELS / "dolos_av_v2_best_model.joblib")

# Save metrics
metrics = {
    "model": best_name,
    "features": len(feature_cols),
    "train_clips": len(train),
    "test_clips": len(test),
    "cv_balanced_accuracy": float(best_score),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_balanced_accuracy": float(balanced_acc)
}

with open(RESULTS / "av_v2_metrics.json", "w") as f:
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
plt.title(f"V2 Audio-Visual Model: {best_name}")
plt.tight_layout()
plt.savefig(RESULTS / "av_v2_confusion_matrix.png", dpi=150)
plt.close()

print("\nV2 MODEL TRAINING COMPLETED!")
print("Saved:", MODELS / "dolos_av_v2_best_model.joblib")
