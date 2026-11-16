
import os
import glob
import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

ROOT = Path("data/DOLOS/bulk_processed")
MANIFESTS = ROOT / "manifests"
OUTPUT = ROOT / "features"

OUTPUT.mkdir(parents=True, exist_ok=True)

def audio_features(path):
    sr, signal = wavfile.read(path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    signal = signal.astype(np.float32)

    if np.issubdtype(signal.dtype, np.integer):
        pass

    peak = np.max(np.abs(signal)) if len(signal) else 0
    if peak > 0:
        signal = signal / peak

    if len(signal) == 0:
        raise ValueError("Empty audio")

    rms = np.sqrt(np.mean(signal ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(signal))) > 0)

    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(len(signal), 1 / sr)

    total = spectrum.sum() + 1e-10
    centroid = np.sum(frequencies * spectrum) / total
    bandwidth = np.sqrt(
        np.sum(((frequencies - centroid) ** 2) * spectrum) / total
    )

    cumulative = np.cumsum(spectrum)
    rolloff = frequencies[
        min(np.searchsorted(cumulative, 0.85 * total), len(frequencies)-1)
    ]

    return {
        "audio_rms": float(rms),
        "audio_zcr": float(zcr),
        "audio_centroid": float(centroid),
        "audio_bandwidth": float(bandwidth),
        "audio_rolloff": float(rolloff),
        "audio_skew": float(skew(signal)),
        "audio_kurtosis": float(kurtosis(signal))
    }


def visual_features(folder):
    files = sorted(
        glob.glob(str(Path(folder) / "*.jpg")) +
        glob.glob(str(Path(folder) / "*.png")) +
        glob.glob(str(Path(folder) / "*.jpeg"))
    )

    if not files:
        raise ValueError("No image frames found")

    indices = np.linspace(
        0, len(files) - 1, min(16, len(files)), dtype=int
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
    face_detector = cv2.CascadeClassifier(cascade_path)

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

        faces = face_detector.detectMultiScale(
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
    print("\nFEATURE EXTRACTION FINISHED")
    print("Output folder:", OUTPUT)

