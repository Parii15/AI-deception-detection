
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

MANIFEST = (
    ROOT
    / "data/DOLOS/bulk_processed/manifests/train.csv"
)

OUTPUT = (
    ROOT
    / "data/DOLOS/bulk_processed/features_face_trial"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT / "facial_trial_balanced10.csv"


# ==========================================
# 2. LOAD OPENCV FACE DETECTORS
# ==========================================

CASCADE_DIR = Path(cv2.data.haarcascades)

face_detector = cv2.CascadeClassifier(
    str(CASCADE_DIR / "haarcascade_frontalface_default.xml")
)

eye_detector = cv2.CascadeClassifier(
    str(CASCADE_DIR / "haarcascade_eye.xml")
)

smile_detector = cv2.CascadeClassifier(
    str(CASCADE_DIR / "haarcascade_smile.xml")
)

if (
    face_detector.empty()
    or eye_detector.empty()
    or smile_detector.empty()
):
    raise RuntimeError("Could not load OpenCV Haar cascades")

print("OpenCV detectors loaded successfully!")


# ==========================================
# 3. SELECT BALANCED SAMPLE
# ==========================================

df = pd.read_csv(MANIFEST)

df = df.drop_duplicates(subset="clip").copy()

df["label"] = df["label"].astype(str).str.strip().str.lower()

# Keep only binary classification labels
df = df[df["label"].isin(["truth", "deception"])]

label_counts = df["label"].value_counts()

print("\nAvailable training labels:")
print(label_counts)

if (
    label_counts.get("truth", 0) < 5
    or label_counts.get("deception", 0) < 5
):
    raise ValueError(
        "Not enough clips for balanced sampling. "
        "Need at least 5 truth and 5 deception clips."
    )

# Select 5 clips from each class
df = (
    df.groupby("label", group_keys=False)
    .sample(n=5, random_state=42)
    .reset_index(drop=True)
)

print("\nSelected balanced sample:")
print(df["label"].value_counts())


# ==========================================
# 4. EXTRACT FACIAL FEATURES
# ==========================================

results = []
errors = []

for _, row in df.iterrows():

    folder = Path(row["frame_folder"])

    if not folder.is_absolute():
        folder = ROOT / folder

    files = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )

    if not files:
        errors.append((row["clip"], "No image frames found"))
        continue

    face_rates = []
    eye_rates = []
    eye_counts = []
    smile_rates = []

    face_sizes = []
    face_centers = []
    mouth_changes = []

    previous_mouth = None

    try:

        for file in files:

            image = cv2.imread(str(file))

            if image is None:
                continue

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            # No face detected
            if len(faces) == 0:

                face_rates.append(0)
                eye_rates.append(0)
                eye_counts.append(0)
                smile_rates.append(0)

                previous_mouth = None

                continue

            # Select largest detected face
            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

            face_rates.append(1)

            face_sizes.append(
                (w * h) / (gray.shape[0] * gray.shape[1])
            )

            face_centers.append(
                (
                    (x + w / 2) / gray.shape[1],
                    (y + h / 2) / gray.shape[0]
                )
            )

            roi = gray[y:y+h, x:x+w]

            rh, rw = roi.shape

            # ==================================
            # EYE FEATURES
            # ==================================

            upper = roi[:int(rh * 0.60), :]

            eyes = eye_detector.detectMultiScale(
                upper,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10)
            )

            eye_counts.append(len(eyes))

            eye_rates.append(
                1 if len(eyes) >= 1 else 0
            )

            # ==================================
            # SMILE FEATURES
            # ==================================

            lower = roi[int(rh * 0.45):, :]

            smiles = smile_detector.detectMultiScale(
                lower,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(15, 10)
            )

            smile_rates.append(
                1 if len(smiles) > 0 else 0
            )

            # ==================================
            # MOUTH REGION VARIATION
            # ==================================

            mouth = cv2.resize(
                lower,
                (64, 32)
            ).astype(np.float32)

            if previous_mouth is not None:

                change = np.mean(
                    np.abs(mouth - previous_mouth)
                )

                mouth_changes.append(float(change))

            previous_mouth = mouth

        # ==================================
        # VALIDATION
        # ==================================

        if not face_rates or sum(face_rates) == 0:
            raise ValueError("No face detected in sampled frames")

        centers = np.array(face_centers)

        # ==================================
        # FINAL FEATURES
        # ==================================

        features = {

            "clip": row["clip"],

            "face_detection_rate": float(
                np.mean(face_rates)
            ),

            "eye_detection_rate": float(
                np.mean(eye_rates)
            ),

            "eye_count_mean": float(
                np.mean(eye_counts)
            ),

            "smile_detection_rate": float(
                np.mean(smile_rates)
            ),

            "face_size_mean": float(
                np.mean(face_sizes)
            ),

            "face_size_std": float(
                np.std(face_sizes)
            ),

            "head_center_motion": (
                float(
                    np.mean(
                        np.linalg.norm(
                            np.diff(centers, axis=0),
                            axis=1
                        )
                    )
                )
                if len(centers) > 1
                else 0.0
            ),

            "mouth_region_change": (
                float(np.mean(mouth_changes))
                if mouth_changes
                else 0.0
            ),

            "label": row["label"]
        }

        results.append(features)

        print(
            f"Processed: {row['clip']} "
            f"({row['label']})"
        )

    except Exception as error:

        errors.append(
            (row["clip"], str(error))
        )

        print(
            f"Failed: {row['clip']} - {error}"
        )


# ==========================================
# 5. SAVE RESULTS
# ==========================================

output = pd.DataFrame(results)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("BALANCED FACIAL TRIAL COMPLETE")
print("================================")

print("Selected clips:", len(df))
print("Successful:", len(results))
print("Failed:", len(errors))

print("\nOutput file:")
print(OUTPUT_FILE)

if not output.empty:

    print("\nExtracted label counts:")
    print(output["label"].value_counts())

    print("\nExtracted features:")
    print(output.to_string(index=False))

if errors:

    print("\nErrors:")
    for clip, error in errors:
        print(clip, ":", error)
