
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

MANIFEST_DIR = ROOT / "data/DOLOS/bulk_processed/manifests"

OUTPUT = (
    ROOT
    / "data/DOLOS/bulk_processed/features_face_full_v1"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

# Process 16 evenly spaced frames per clip
FRAMES_PER_CLIP = 16

# ==========================================
# 2. LOAD OPENCV DETECTORS
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

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded")

if eye_detector.empty():
    raise RuntimeError("Eye detector could not be loaded")

if smile_detector.empty():
    raise RuntimeError("Smile detector could not be loaded")

print("OpenCV facial detectors loaded successfully!")


# ==========================================
# 3. RESOLVE FRAME FOLDER
# ==========================================

def resolve_folder(value):

    folder = Path(str(value))

    if not folder.is_absolute():
        folder = ROOT / folder

    return folder


# ==========================================
# 4. EXTRACT FEATURES FROM ONE CLIP
# ==========================================

def extract_features(row):

    folder = resolve_folder(row["frame_folder"])

    files = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )

    if not files:
        raise ValueError(f"No frames found: {folder}")

    # Select evenly spaced frames
    indices = np.linspace(
        0,
        len(files) - 1,
        min(FRAMES_PER_CLIP, len(files)),
        dtype=int
    )

    selected_files = [files[i] for i in indices]

    face_rates = []
    eye_rates = []
    eye_counts = []
    smile_rates = []

    face_sizes = []
    face_centers = []
    mouth_changes = []

    previous_mouth = None

    for file in selected_files:

        image = cv2.imread(str(file))

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:

            face_rates.append(0)
            eye_rates.append(0)
            eye_counts.append(0)
            smile_rates.append(0)

            previous_mouth = None
            continue

        # Largest detected face
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

        # ------------------------------
        # EYE FEATURES
        # ------------------------------

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

        # ------------------------------
        # SMILE FEATURES
        # ------------------------------

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

        # ------------------------------
        # MOUTH REGION VARIATION
        # ------------------------------

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

    if not face_rates or sum(face_rates) == 0:
        raise ValueError("No face detected in selected frames")

    centers = np.array(face_centers)

    # ------------------------------
    # FINAL CLIP FEATURES
    # ------------------------------

    return {

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

        "label": str(row["label"]).strip().lower()
    }


# ==========================================
# 5. PROCESS TRAIN / TEST
# ==========================================

def process_split(filename):

    manifest = MANIFEST_DIR / filename

    df = pd.read_csv(manifest)

    df = df.drop_duplicates(subset="clip").copy()

    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
    )

    df = df[
        df["label"].isin(["truth", "deception"])
    ].reset_index(drop=True)

    results = []
    errors = []

    print(f"\nStarting: {filename}")
    print("Total clips:", len(df))

    for i, row in df.iterrows():

        try:

            features = extract_features(row)

            results.append(features)

        except Exception as error:

            errors.append({
                "clip": row["clip"],
                "error": str(error)
            })

        if (i + 1) % 50 == 0:

            print(
                f"{filename}: {i+1}/{len(df)} processed | "
                f"Successful: {len(results)} | "
                f"Failed: {len(errors)}"
            )

    # Save features in a new folder
    output_file = OUTPUT / filename.replace(
        ".csv",
        "_facial_features.csv"
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    # Save error report separately
    error_file = OUTPUT / filename.replace(
        ".csv",
        "_errors.csv"
    )

    pd.DataFrame(errors).to_csv(
        error_file,
        index=False
    )

    print(f"\n{filename} COMPLETE")
    print("Successful:", len(results))
    print("Failed:", len(errors))
    print("Saved:", output_file)

    if results:

        result_df = pd.DataFrame(results)

        print("\nLabel counts:")
        print(result_df["label"].value_counts())


# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":

    process_split("train.csv")

    process_split("test.csv")

    print("\n================================")
    print("FULL FACIAL FEATURE EXTRACTION DONE")
    print("================================")

    print("Output folder:", OUTPUT)
    