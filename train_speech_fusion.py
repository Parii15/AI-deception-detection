"""
Experimental Audio-Visual-Speech Fusion Model
Dataset: DOLOS

Uses:
- 49 audio features
- 5 visual features
- 8 facial features
- 6 speech features extracted using Faster-Whisper

Total: 68 model input features.

This is a research prototype, NOT a scientifically validated lie detector.
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from faster_whisper import WhisperModel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "DOLOS" / "bulk_processed"

FUSION_DIR = DATA / "features_face_fusion_v2"
MANIFEST_DIR = DATA / "manifests"

OUT_DIR = ROOT / "models" / "speech_fusion_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SPEECH FEATURE SETTINGS
# ==========================================

MODEL_NAME = "base.en"

FILLERS = {
    "um",
    "uh",
    "uhm",
    "erm",
    "er",
    "hmm",
    "umm",
    "uhh"
}

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)

SPEECH_FEATURES = [
    "speech_pause_count",
    "speech_long_pause_count",
    "speech_pause_total_seconds",
    "speech_filler_count",
    "speech_repeated_word_count",
    "speech_rate_wpm"
]


# ==========================================
# 3. LABEL NORMALIZATION
# ==========================================

def normalize_label(value):

    label = str(value).strip().lower()

    if label == "lie":
        label = "deception"

    if label not in ["truth", "deception"]:
        raise ValueError(
            f"Unexpected label found: {value}"
        )

    return label


# ==========================================
# 4. AUDIO PATH
# ==========================================

def resolve_audio_path(value):

    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path


# ==========================================
# 5. SPEECH FEATURE EXTRACTION
# ==========================================

def extract_speech_features(whisper, audio_path):

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    segments, info = whisper.transcribe(
        str(audio_path),
        language="en",
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False
    )

    words = []

    for segment in segments:

        if segment.words:

            for word in segment.words:

                tokens = WORD_PATTERN.findall(
                    word.word.lower()
                )

                if tokens:

                    words.append(
                        (
                            tokens[0],
                            float(word.start),
                            float(word.end)
                        )
                    )

    words.sort(key=lambda x: x[1])

    # --------------------------------------
    # Pause analysis
    # --------------------------------------

    pause_gaps = []

    for previous, current in zip(
        words,
        words[1:]
    ):

        gap = max(
            0.0,
            current[1] - previous[2]
        )

        if gap >= 0.40:
            pause_gaps.append(gap)

    pause_count = len(pause_gaps)

    long_pause_count = sum(
        gap >= 1.0
        for gap in pause_gaps
    )

    total_pause_duration = sum(
        pause_gaps
    )

    # --------------------------------------
    # Filler word analysis
    # --------------------------------------

    tokens = [
        word[0]
        for word in words
    ]

    filler_count = sum(
        token in FILLERS
        for token in tokens
    )

    # --------------------------------------
    # Repeated word analysis
    # --------------------------------------

    repeated_count = sum(
        1
        for first, second in zip(
            tokens,
            tokens[1:]
        )
        if first == second
    )

    # --------------------------------------
    # Speech rate
    # --------------------------------------

    if words:

        speech_duration = max(
            words[-1][2] - words[0][1],
            0.000001
        )

        speech_rate = (
            len(words) / speech_duration
        ) * 60

    else:

        speech_rate = 0.0

    return {

        "speech_pause_count":
            float(pause_count),

        "speech_long_pause_count":
            float(long_pause_count),

        "speech_pause_total_seconds":
            float(total_pause_duration),

        "speech_filler_count":
            float(filler_count),

        "speech_repeated_word_count":
            float(repeated_count),

        "speech_rate_wpm":
            float(speech_rate)
    }


# ==========================================
# 6. LOAD TRAIN / TEST DATA
# ==========================================

def load_split(split, whisper):

    fusion_path = (
        FUSION_DIR /
        f"{split}_audio_facial.csv"
    )

    manifest_path = (
        MANIFEST_DIR /
        f"{split}.csv"
    )

    if not fusion_path.exists():
        raise FileNotFoundError(fusion_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    fusion = pd.read_csv(fusion_path)

    manifest = pd.read_csv(manifest_path)

    fusion = fusion.drop_duplicates(
        subset="clip"
    )

    manifest = manifest.drop_duplicates(
        subset="clip"
    )

    if "audio" not in manifest.columns:
        raise ValueError(
            "Manifest does not contain an 'audio' column."
        )

    manifest_columns = ["clip", "audio"]

    if "youtube_id" in manifest.columns:
        manifest_columns.append("youtube_id")

    merged = fusion.merge(
        manifest[manifest_columns],
        on="clip",
        how="left",
        validate="one_to_one"
    )

    if merged["audio"].isna().any():

        raise ValueError(
            "Some clips have missing audio paths."
        )

    # --------------------------------------
    # Speech feature cache
    # --------------------------------------

    cache_path = (
        OUT_DIR /
        f"{split}_speech_features.csv"
    )

    cache = {}

    if cache_path.exists():

        cached = pd.read_csv(cache_path)

        for _, row in cached.iterrows():

            cache[str(row["clip"])] = {
                feature: float(row[feature])
                for feature in SPEECH_FEATURES
                if feature in cached.columns
                and pd.notna(row[feature])
            }

    results = []

    failures = []

    total = len(merged)

    for index, row in merged.iterrows():

        clip = str(row["clip"])

        if (
            clip in cache
            and all(
                feature in cache[clip]
                for feature in SPEECH_FEATURES
            )
        ):

            features = cache[clip]

        else:

            audio_path = resolve_audio_path(
                row["audio"]
            )

            try:

                features = extract_speech_features(
                    whisper,
                    audio_path
                )

                cache[clip] = features

            except Exception as error:

                failures.append(
                    (clip, str(error))
                )

                continue

        results.append({
            "clip": clip,
            **features
        })

        # Save progress every 25 clips
        if (index + 1) % 25 == 0:

            pd.DataFrame([
                {
                    "clip": key,
                    **value
                }
                for key, value in cache.items()
            ]).to_csv(
                cache_path,
                index=False
            )

            print(
                f"{split}: {index + 1}/{total} clips processed"
            )

    # Save final cache
    pd.DataFrame([
        {
            "clip": key,
            **value
        }
        for key, value in cache.items()
    ]).to_csv(
        cache_path,
        index=False
    )

    if failures:

        print(
            f"\n{len(failures)} clips could not be processed."
        )

        for clip, error in failures[:10]:

            print(clip, error)

    speech_df = pd.DataFrame(results)

    if speech_df.empty:

        raise RuntimeError(
            f"No speech features extracted for {split}."
        )

    final_data = fusion.merge(
        speech_df,
        on="clip",
        how="inner",
        validate="one_to_one"
    )

    final_data["label"] = (
        final_data["label"].apply(normalize_label)
    )

    return final_data, merged


# ==========================================
# 7. MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("\n================================")
    print(" AUDIO-VISUAL-SPEECH MODEL")
    print("================================\n")

    print("Loading Faster-Whisper model...")

    whisper = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8"
    )

    # --------------------------------------
    # Load training data
    # --------------------------------------

    print("\nExtracting TRAIN speech features...")

    train, train_manifest = load_split(
        "train",
        whisper
    )

    # --------------------------------------
    # Load testing data
    # --------------------------------------

    print("\nExtracting TEST speech features...")

    test, test_manifest = load_split(
        "test",
        whisper
    )

    # --------------------------------------
    # Model input features
    # --------------------------------------

    excluded = {
        "clip",
        "label",
        "audio",
        "youtube_id"
    }

    base_features = [
        column
        for column in train.columns
        if column not in excluded
        and column not in SPEECH_FEATURES
    ]

    feature_columns = (
        base_features + SPEECH_FEATURES
    )

    missing_features = (
        set(feature_columns) - set(test.columns)
    )

    if missing_features:

        raise ValueError(
            f"Missing test features: {missing_features}"
        )

    X_train = train[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = test[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y_train = train["label"]

    y_test = test["label"]

    print("\nTraining samples:", len(X_train))

    print("Testing samples:", len(X_test))

    print("Total model input features:", len(feature_columns))

    print("\nFeatures used by model:")

    for feature in feature_columns:
        print("-", feature)

    # --------------------------------------
    # Model pipeline
    # --------------------------------------

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="linear",
                C=0.01,
                class_weight="balanced"
            )
        )

    ])

    # --------------------------------------
    # Grouped cross validation
    # --------------------------------------

    if "youtube_id" in train_manifest.columns:

        source_map = (
            train_manifest
            .drop_duplicates("clip")
            .set_index("clip")["youtube_id"]
        )

        groups = train["clip"].map(
            source_map
        )

        if (
            groups.notna().all()
            and groups.nunique() >= 5
        ):

            print(
                "\nRunning 5-fold source-grouped validation..."
            )

            cv = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                groups=groups,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=-1
            )

            print(
                "\nGrouped CV scores:",
                np.round(scores, 4)
            )

            print(
                "Mean balanced accuracy:",
                round(float(scores.mean()), 4)
            )

        else:

            print(
                "\nGrouped CV skipped: insufficient source IDs."
            )

    else:

        print(
            "\nGrouped CV skipped: youtube_id unavailable."
        )

    # --------------------------------------
    # Train final model
    # --------------------------------------

    print("\nTraining final model...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------
    # Held-out test evaluation
    # --------------------------------------

    predictions = model.predict(
        X_test
    )

    print("\n================================")
    print(" HELD-OUT TEST RESULTS")
    print("================================")

    print(
        "\nAccuracy:",
        round(
            accuracy_score(
                y_test,
                predictions
            ),
            4
        )
    )

    print(
        "Balanced accuracy:",
        round(
            balanced_accuracy_score(
                y_test,
                predictions
            ),
            4
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["deception", "truth"]
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["deception", "truth"],
            zero_division=0
        )
    )

    print("\nPrediction counts:")

    print(
        pd.Series(predictions).value_counts()
    )

    # --------------------------------------
    # Save new model
    # --------------------------------------

    model_bundle = {

        "model": model,

        "feature_columns": feature_columns,

        "selected_model": "Linear SVM",

        "labels": [
            "deception",
            "truth"
        ],

        "speech_features": SPEECH_FEATURES,

        "note": (
            "Experimental DOLOS audio-visual-speech "
            "classifier. Not a scientifically validated lie detector."
        )
    }

    model_path = (
        OUT_DIR /
        "dolos_audio_visual_speech_model.joblib"
    )

    joblib.dump(
        model_bundle,
        model_path
    )

    print("\n================================")

    print("NEW MODEL SAVED SUCCESSFULLY")

    print(model_path)

    print("\nOriginal fusion model was not overwritten.")

    print("================================")


# ==========================================
# 8. RUN
# ==========================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "\nERROR:",
            error,
            file=sys.stderr
        )

        raise