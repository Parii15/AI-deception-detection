
from pathlib import Path
import pandas as pd

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
    raise ValueError("Missing source IDs")

cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(score_func=f_classif)),
    ("scale", StandardScaler()),
    ("model", SVC(kernel="linear", class_weight="balanced"))
])

search = GridSearchCV(
    pipeline,
    {
        "select__k": [10, 20, 30, 40, "all"],
        "model__C": [0.01, 0.1, 1, 10]
    },
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X, y, groups=groups)

results = pd.DataFrame(search.cv_results_)
results = results[
    ["param_select__k", "param_model__C", "mean_test_score"]
].sort_values("mean_test_score", ascending=False)

print("\nFEATURE SELECTION RESULTS")
print(results.to_string(index=False))

print("\nBEST FEATURE COUNT:", search.best_params_["select__k"])
print("BEST CV BALANCED ACCURACY:", round(search.best_score_, 4))
print("BEST PARAMETERS:", search.best_params_)
