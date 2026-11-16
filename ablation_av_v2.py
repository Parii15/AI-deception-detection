
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold
)

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "DOLOS" / "bulk_processed"
FEATURES = DATA / "features_v2"
MANIFESTS = DATA / "manifests"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

# Load training data only. Test data is intentionally untouched.
train = pd.read_csv(FEATURES / "train_features.csv")
manifest = pd.read_csv(MANIFESTS / "train.csv")

# Remove duplicate clips consistently
train = train.drop_duplicates(
    subset="clip", keep="first"
).reset_index(drop=True)

manifest = manifest.drop_duplicates(
    subset="clip", keep="first"
).reset_index(drop=True)

# Identify feature groups
audio_cols = [
    c for c in train.columns
    if c.startswith("audio_")
]

mfcc_cols = [
    c for c in train.columns
    if c.startswith("mfcc_")
]

visual_cols = [
    c for c in train.columns
    if c.startswith("visual_")
]

feature_groups = {
    "Audio + MFCC": audio_cols + mfcc_cols,
    "Visual only": visual_cols,
    "Audio + MFCC + Visual": audio_cols + mfcc_cols + visual_cols,
}

# Labels and source groups
y = train["label"].str.strip().str.lower()

group_map = manifest.set_index("clip")["youtube_id"]
groups = train["clip"].map(group_map)

if groups.isna().any():
    raise ValueError("Some clips have missing YouTube source IDs")

if groups.nunique() < 5:
    raise ValueError("Not enough unique source groups for 5-fold CV")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

param_grid = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["rbf", "linear"]
}

results = []

print("Training clips:", len(train))
print("Unique sources:", groups.nunique())
print("Audio features:", len(audio_cols))
print("MFCC features:", len(mfcc_cols))
print("Visual features:", len(visual_cols))

for name, columns in feature_groups.items():

    if not columns:
        print("Skipping empty feature group:", name)
        continue

    print(f"\nRunning experiment: {name}")
    print("Features:", len(columns))

    X = train[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(class_weight="balanced"))
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    search.fit(X, y, groups=groups)

    results.append({
        "feature_group": name,
        "feature_count": len(columns),
        "cv_balanced_accuracy": search.best_score_,
        "best_parameters": str(search.best_params_)
    })

    print(
        "CV balanced accuracy:",
        round(search.best_score_, 4)
    )
    print("Best parameters:", search.best_params_)

# Save comparison
comparison = pd.DataFrame(results).sort_values(
    "cv_balanced_accuracy",
    ascending=False
)

comparison.to_csv(
    RESULTS / "av_v2_feature_ablation.csv",
    index=False
)

print("\n===== FEATURE ABLATION RESULTS =====")
print(comparison.to_string(index=False))

print("\nSaved to:", RESULTS / "av_v2_feature_ablation.csv")
print("Ablation experiment completed.")
