
from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "DOLOS" / "bulk_processed"
FEATURES = DATA / "features_v2"
MANIFESTS = DATA / "manifests"
MODELS = ROOT / "models"

train = pd.read_csv(FEATURES / "train_features.csv")
manifest = pd.read_csv(MANIFESTS / "train.csv")

train = train.drop_duplicates(
    subset="clip", keep="first"
).reset_index(drop=True)

manifest = manifest.drop_duplicates(
    subset="clip", keep="first"
).reset_index(drop=True)

# Select audio and MFCC features only
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

if len(columns) != 49:
    raise ValueError(f"Expected 49 features, got {len(columns)}")

# Train selected model using all training clips
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(C=0.1, kernel="linear", class_weight="balanced"))
])

model.fit(X, y)

MODELS.mkdir(exist_ok=True)

joblib.dump({
    "model": model,
    "feature_columns": columns,
    "model_name": "Audio + MFCC Linear SVM",
    "cv_balanced_accuracy": 0.6245
}, MODELS / "dolos_audio_mfcc_model.joblib")

print("Audio + MFCC model trained successfully!")
print("Training clips:", len(train))
print("Features:", len(columns))
print("Saved:", MODELS / "dolos_audio_mfcc_model.joblib")

