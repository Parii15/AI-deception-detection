
from pathlib import Path
import glob

import cv2
import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

MANIFESTS = ROOT / "data" / "DOLOS" / "bulk_processed" / "manifests"
OUTPUT = ROOT / "data" / "DOLOS" / "bulk_processed" / "features_v2"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ---------------- AUDIO FEATURES ----------------

def audio_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)

    if len(y) < 1024:
        raise ValueError("Audio too short")

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)

    # Pitch
    pitch = librosa.yin(
        y, fmin=65, fmax=400,
        sr=sr, frame_length=1024
    )

    # Voice energy
    rms = librosa.feature.rms(
        y=y, frame_length=1024, hop_length=256
    )[0]

    # Pause ratio (approximate low-energy frames)
    threshold = max(float(np.max(rms)) * 0.10, 1e-5)
    pause_ratio = float(np.mean(rms < threshold))

    # Other audio features
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    voiced_pitch = pitch[np.isfinite(pitch) & (pitch > 0)]

    features = {
        "audio_rms_mean": float(np.mean(rms)),
        "audio_rms_std": float(np.std(rms)),
        "audio_pause_ratio": pause_ratio,
        "audio_zcr_mean": float(np.mean(zcr)),
        "audio_centroid_mean": float(np.mean(centroid)),
        "audio_centroid_std": float(np.std(centroid)),
        "audio_bandwidth_mean": float(np.mean(bandwidth)),
        "audio_rolloff_mean": float(np.mean(rolloff)),
        "audio_pitch_median": (
            float(np.median(voiced_pitch))
            if len(voiced_pitch) else 0.0
        ),
        "audio_pitch_std": (
            float(np.std(voiced_pitch))
            if len(voiced_pitch) else 0.0
        )
    }

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(delta[i]))

    return features


# ---------------- VISUAL FEATURES ----------------

def visual_features(folder):
    files = sorted(
        glob.glob(str(Path(folder) / "*.jpg")) +
        glob.glob(str(Path(folder) / "*.png")) +
        glob.glob(str(Path(folder) / "*.jpeg"))
    )

    if not files:
        raise ValueError("No image frames found")

    indices = np.linspace(
        0, len(files) - 1,
        min(16, len(files)), dtype=int
    )

    brightness = []
    contrast = []
    edges = []
    motion = []
    face_counts = []

    previous = None

    cascade_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded")

    for index in indices:
        image = cv2.imread(files[index])

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        brightness.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))

        edge = cv2.Canny(gray, 80, 160)
        edges.append(float(np.mean(edge > 0)))

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4
        )
        face_counts.append(len(faces))

        if previous is not None:
            motion.append(float(
                np.mean(cv2.absdiff(gray, previous))
            ))

        previous = gray

    if not brightness:
        raise ValueError("Could not read frames")

    return {
        "visual_brightness": float(np.mean(brightness)),
        "visual_contrast": float(np.mean(contrast)),
        "visual_edges": float(np.mean(edges)),
        "visual_motion": float(np.mean(motion)) if motion else 0.0,
        "visual_face_count": float(np.mean(face_counts))
    }


# ---------------- PROCESS DATA ----------------

def process_split(filename):
    df = pd.read_csv(MANIFESTS / filename)

    results = []
    errors = []

    for i, row in df.iterrows():
        try:
            features = {}
            features.update(audio_features(row["audio"]))
            features.update(visual_features(row["frame_folder"]))

            features["clip"] = row["clip"]
            features["label"] = row["label"].strip().lower()

            results.append(features)

        except Exception as error:
            errors.append((row["clip"], str(error)))

        if (i + 1) % 50 == 0:
            print(f"{filename}: {i+1}/{len(df)} processed")

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT / filename.replace(".csv", "_features.csv"),
        index=False
    )

    print("\n", filename, "COMPLETE")
    print("Successful:", len(results))
    print("Failed:", len(errors))

    if errors:
        print("First 10 errors:", errors[:10])


if __name__ == "__main__":
    process_split("train.csv")
    process_split("test.csv")

    print("\nV2 FEATURE EXTRACTION FINISHED")
    print("Output folder:", OUTPUT)
