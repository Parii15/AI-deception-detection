
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "DOLOS" / "bulk_processed"
FEATURES = DATA / "features_v2"
MANIFESTS = DATA / "manifests"
MODELS = ROOT / "models"

train = pd.read_csv(FEATURES / "train_features.csv")
manifest = pd.read_csv(MANIFESTS / "train.csv")

train = train.drop_duplicates("clip").reset_index(drop=True)
manifest = manifest.drop_duplicates("clip").reset_index(drop=True)

columns = [
    c for c in train.columns
    if c.startswith("audio_") or c.startswith("mfcc_")
]

X = train[columns].apply(pd.to_numeric, errors="coerce")
y = train["label"].str.strip().str.lower()

groups = train["clip"].map(
    manifest.set_index("clip")["youtube_id"]
)

if groups.isna().any():
    raise ValueError("Missing YouTube source IDs")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(score_func=f_classif)),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear", class_weight="balanced"))
])

search = GridSearchCV(
    pipeline,
    {
        "select__k": [40],
        "model__C": [0.1]
    },
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("Training selected 40-feature model...")
search.fit(X, y, groups=groups)

best_model = search.best_estimator_
selected_mask = best_model.named_steps["select"].get_support()
selected_features = [
    c for c, keep in zip(columns, selected_mask) if keep
]

MODELS.mkdir(exist_ok=True)

joblib.dump({
    "model": best_model,
    "feature_columns": columns,
    "selected_features": selected_features,
    "model_name": "Selected 40 Audio + MFCC Linear SVM",
    "cv_balanced_accuracy": search.best_score_
}, MODELS / "dolos_audio_mfcc_selected40.joblib")

print("\nMODEL TRAINED SUCCESSFULLY!")
print("Training clips:", len(train))
print("Original features:", len(columns))
print("Selected features:", len(selected_features))
print("CV balanced accuracy:", round(search.best_score_, 4))
print("Saved: models/dolos_audio_mfcc_selected40.joblib")
print("\nSelected feature names:")
print(selected_features)
