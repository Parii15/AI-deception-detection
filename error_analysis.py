
import os
import pandas as pd

# ==========================================
# PROJECT PATHS
# ==========================================

BASE = os.getcwd()

PREDICTION_PATH = os.path.join(
    BASE,
    "results",
    "final_test_mfcc_v1",
    "predictions.csv"
)

TRAIN_PATH = os.path.join(
    BASE,
    "data",
    "DOLOS",
    "bulk_processed",
    "manifests",
    "train.csv"
)

TEST_PATH = os.path.join(
    BASE,
    "data",
    "DOLOS",
    "bulk_processed",
    "manifests",
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "final_test_mfcc_v1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# PART 1: EPISODE-WISE ERROR ANALYSIS
# ==========================================

print("\n====================================")
print("EPISODE-WISE ERROR ANALYSIS")
print("====================================")

df = pd.read_csv(PREDICTION_PATH)

required_prediction_columns = {
    "clip", "actual", "predicted"
}

if not required_prediction_columns.issubset(df.columns):
    raise ValueError(
        "Predictions CSV mein required columns missing hain."
    )

# Extract episode name from clip name
df["episode"] = df["clip"].astype(str).str.extract(
    r"^(.+?_WILTY_EP\d+)"
)

df["correct"] = df["actual"] == df["predicted"]

# Calculate episode-wise performance
summary = (
    df.groupby("episode")
    .agg(
        total_clips=("clip", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        wrong=("correct", lambda x: (~x).sum())
    )
    .reset_index()
)

summary["accuracy"] = (
    summary["accuracy"] * 100
).round(2)

summary = summary.sort_values(
    "accuracy",
    ascending=True
)

print("\n===== EPISODE-WISE ACCURACY =====")
print(summary.to_string(index=False))

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "episode_error_summary.csv"
    ),
    index=False
)

print("\nEpisode report saved.")


# ==========================================
# PART 2: TRAIN-TEST SOURCE LEAKAGE CHECK
# ==========================================

print("\n====================================")
print("TRAIN-TEST SOURCE LEAKAGE CHECK")
print("====================================")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_columns = {"clip", "youtube_id"}

if not required_columns.issubset(train.columns):
    raise ValueError(
        "Train manifest mein clip/youtube_id missing hai."
    )

if not required_columns.issubset(test.columns):
    raise ValueError(
        "Test manifest mein clip/youtube_id missing hai."
    )

# Find unique YouTube sources
train_sources = set(
    train["youtube_id"].dropna().astype(str)
)

test_sources = set(
    test["youtube_id"].dropna().astype(str)
)

# Find common sources
common_sources = train_sources.intersection(
    test_sources
)

print("\nTrain clips:", len(train))
print("Test clips:", len(test))

print(
    "Unique train sources:",
    len(train_sources)
)

print(
    "Unique test sources:",
    len(test_sources)
)

print(
    "Common sources:",
    len(common_sources)
)

if len(common_sources) > 0:

    print("\nWARNING: Common YouTube sources found!")

    for source in sorted(common_sources):
        print(source)

    common_train = train[
        train["youtube_id"].astype(str).isin(common_sources)
    ].copy()

    common_test = test[
        test["youtube_id"].astype(str).isin(common_sources)
    ].copy()

    common_train["split"] = "train"
    common_test["split"] = "test"

    common_details = pd.concat(
        [common_train, common_test],
        ignore_index=True
    )

    common_details.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "common_sources.csv"
        ),
        index=False
    )

else:
    print("\nPASS: No common YouTube sources found.")


# ==========================================
# PART 3: TEST EPISODE SOURCE MAPPING
# ==========================================

print("\n====================================")
print("TEST EPISODE SOURCE MAPPING")
print("====================================")

test["episode"] = test["clip"].astype(str).str.extract(
    r"^(.+?_WILTY_EP\d+)"
)

episode_sources = (
    test.groupby("episode")
    .agg(
        clips=("clip", "count"),
        youtube_sources=("youtube_id", "nunique")
    )
    .reset_index()
)

print(
    episode_sources.to_string(index=False)
)

episode_sources.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "episode_source_mapping.csv"
    ),
    index=False
)


# ==========================================
# PART 4: COMPLETION
# ==========================================

print("\n====================================")
print("ALL ANALYSIS COMPLETED")
print("====================================")

print("Results folder:")
print(OUTPUT_DIR)

print("\nGenerated reports:")
print("- episode_error_summary.csv")
print("- episode_source_mapping.csv")

if len(common_sources) > 0:
    print("- common_sources.csv")

print("\nAnalysis finished successfully!")
