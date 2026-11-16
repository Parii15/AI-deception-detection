
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

AUDIO_DIR = ROOT / "data/DOLOS/bulk_processed/features_v2"

FACE_DIR = ROOT / "data/DOLOS/bulk_processed/features_face_full_v1"

# New output folder: old files remain untouched
OUTPUT_DIR = ROOT / "data/DOLOS/bulk_processed/features_face_fusion_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_duplicates(df, dataset_name, split):

    # Find all duplicate clip IDs
    duplicate_rows = df[df.duplicated("clip", keep=False)]

    removed = 0

    if not duplicate_rows.empty:

        print(
            f"{dataset_name}: Found "
            f"{len(duplicate_rows)} duplicate rows"
        )

        for clip, group in duplicate_rows.groupby("clip"):

            # Compare every column except clip
            unique_versions = group.drop(
                columns=["clip"]
            ).drop_duplicates()

            # Stop if duplicate rows contain different data
            if len(unique_versions) > 1:
                raise ValueError(
                    f"Conflicting duplicate {dataset_name} "
                    f"features in {split}: {clip}"
                )

        # All duplicates are identical, so keep one copy
        before = len(df)

        df = df.drop_duplicates(
            subset=["clip"],
            keep="first"
        ).copy()

        removed = before - len(df)

    print(
        f"{dataset_name}: Removed {removed} "
        "identical duplicate rows"
    )

    return df


def merge_split(split):

    print(f"\n{'=' * 45}")
    print(f"PROCESSING {split.upper()}")
    print(f"{'=' * 45}")

    audio_path = AUDIO_DIR / f"{split}_features.csv"
    face_path = FACE_DIR / f"{split}_facial_features.csv"

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    if not face_path.exists():
        raise FileNotFoundError(face_path)

    # Load feature files
    audio = pd.read_csv(audio_path)
    face = pd.read_csv(face_path)

    print("Original audio rows:", len(audio))
    print("Original facial rows:", len(face))

    # Normalize clip IDs and labels
    for df in (audio, face):

        required = {"clip", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                "Missing required clip or label column"
            )

        df["clip"] = (
            df["clip"].astype(str).str.strip()
        )

        df["label"] = (
            df["label"].astype(str).str.strip().str.lower()
        )

    # Check for empty clip IDs
    if audio["clip"].eq("").any() or face["clip"].eq("").any():
        raise ValueError("Empty clip IDs found")

    # Safely remove identical duplicates
    audio = clean_duplicates(audio, "Audio", split)
    face = clean_duplicates(face, "Facial", split)

    # Check that labels are consistent for matching clips
    common = set(audio["clip"]) & set(face["clip"])

    audio_labels = audio.set_index("clip")["label"]
    face_labels = face.set_index("clip")["label"]

    mismatches = [
        clip for clip in common
        if audio_labels.loc[clip] != face_labels.loc[clip]
    ]

    if mismatches:
        raise ValueError(
            f"Label mismatches in {split}: {mismatches[:10]}"
        )

    # Merge only facial features with audio features
    merged = audio.merge(
        face,
        on="clip",
        how="inner",
        suffixes=("", "_face"),
        validate="one_to_one"
    )

    # Remove duplicate facial label column
    if "label_face" in merged.columns:
        merged = merged.drop(columns=["label_face"])

    # Feature columns
    feature_cols = [
        col for col in merged.columns
        if col not in ["clip", "label"]
    ]

    # Check missing values
    missing = merged[feature_cols].isna().sum().sum()

    # Check duplicate clips after merging
    duplicate_count = merged["clip"].duplicated().sum()

    if duplicate_count:
        raise ValueError(
            f"Duplicate clips remain after merging: {duplicate_count}"
        )

    # Save to new output folder
    output_path = OUTPUT_DIR / f"{split}_audio_facial.csv"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    merged.to_csv(output_path, index=False)

    # Final report
    print(f"\n{split.upper()} MERGE RESULTS")
    print("-----------------------------------")
    print("Audio clips after cleaning:", len(audio))
    print("Facial clips after cleaning:", len(face))
    print("Matched clips:", len(merged))
    print("Unmatched audio clips:", len(audio) - len(merged))
    print("Unmatched facial clips:", len(face) - len(merged))
    print("Total features:", len(feature_cols))
    print("Missing feature values:", missing)
    print("Duplicate clips after merge:", duplicate_count)
    print("Label distribution:")
    print(merged["label"].value_counts())
    print("Saved:", output_path)

    return len(merged)


if __name__ == "__main__":

    train_count = merge_split("train")
    test_count = merge_split("test")

    print("\n" + "=" * 45)
    print("AUDIO + FACIAL FEATURE MERGE COMPLETED")
    print("=" * 45)

    print("Final training clips:", train_count)
    print("Final testing clips:", test_count)
    print("Output folder:", OUTPUT_DIR)
