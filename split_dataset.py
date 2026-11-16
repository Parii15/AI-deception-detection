
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

source = Path(
    "data/DOLOS/bulk_processed/manifests/matched_clips.csv"
)

output = Path("data/DOLOS/bulk_processed/manifests")
df = pd.read_csv(source)

# Group by original YouTube video to prevent leakage
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        y=df["label"],
        groups=df["youtube_id"]
    )
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

# Verify no source video overlaps
overlap = set(train["youtube_id"]) & set(test["youtube_id"])

assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"

train.to_csv(output / "train.csv", index=False)
test.to_csv(output / "test.csv", index=False)

print("\n--- DATASET SPLIT COMPLETE ---")
print("Total clips:", len(df))
print("Training clips:", len(train))
print("Testing clips:", len(test))

print("\nTraining labels:")
print(train["label"].value_counts())

print("\nTesting labels:")
print(test["label"].value_counts())

print("\nTrain source videos:", train["youtube_id"].nunique())
print("Test source videos:", test["youtube_id"].nunique())
print("Overlapping source videos:", len(overlap))
