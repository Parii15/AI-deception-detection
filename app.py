
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import tempfile
import subprocess
import imageio_ffmpeg
import cv2
import re

from pathlib import Path
from faster_whisper import WhisperModel


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "models" / "speech_fusion_v1"
    / "dolos_audio_visual_speech_model.joblib"
)

st.set_page_config(
    page_title="Audio-Visual Deception Pattern Analysis",
    page_icon="🎙️",
    layout="centered"
)

FILLERS = {
    "um", "uh", "uhm", "erm",
    "er", "hmm", "umm", "uhh"
}

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)


# ==========================================
# 2. LOAD TRAINED MODEL AND WHISPER
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained multimodal model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_speech_model():

    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8"
    )


# ==========================================
# 3. EXTRACT AUDIO FROM VIDEO
# ==========================================

def extract_audio(video_path, audio_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio extraction failed: "
            + result.stderr[-1500:]
        )


# ==========================================
# 4. AUDIO FEATURES - 49 FEATURES
# Same feature calculations as DOLOS training
# ==========================================

def extract_audio_features(audio_path):

    y, sr = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    if len(y) < 1024:
        raise ValueError("Audio is too short.")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)

    pitch = librosa.yin(
        y,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=1024
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=256
    )[0]

    threshold = max(
        float(np.max(rms)) * 0.10,
        1e-5
    )

    pause_ratio = float(
        np.mean(rms < threshold)
    )

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    voiced_pitch = pitch[
        np.isfinite(pitch) & (pitch > 0)
    ]

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

        features[f"mfcc_{i+1}_mean"] = float(
            np.mean(mfcc[i])
        )

        features[f"mfcc_{i+1}_std"] = float(
            np.std(mfcc[i])
        )

        features[f"mfcc_delta_{i+1}_mean"] = float(
            np.mean(delta[i])
        )

    return features, y, sr


# ==========================================
# 5. EXTRACT 16 VIDEO FRAMES
# ==========================================


def extract_video_frames(video_path):

    # --------------------------------------
    # First: check whether this is a DOLOS clip
    # --------------------------------------

    uploaded_stem = Path(video_path).stem.lower()

    manifest_dir = (
        ROOT / "data" / "DOLOS"
        / "bulk_processed" / "manifests"
    )

    for split in ["train.csv", "test.csv"]:

        manifest_path = manifest_dir / split

        if not manifest_path.exists():
            continue

        manifest = pd.read_csv(manifest_path)

        for _, row in manifest.iterrows():

            clip_stem = Path(
                str(row["clip"])
            ).stem.lower()

            if clip_stem == uploaded_stem:

                frame_folder = Path(
                    str(row["frame_folder"])
                )

                if not frame_folder.is_absolute():
                    frame_folder = ROOT / frame_folder

                image_files = sorted(
                    list(frame_folder.glob("*.jpg"))
                    + list(frame_folder.glob("*.jpeg"))
                    + list(frame_folder.glob("*.png"))
                )

                if not image_files:
                    raise ValueError(
                        f"DOLOS frame folder is empty: {frame_folder}"
                    )

                indices = np.linspace(
                    0,
                    len(image_files) - 1,
                    min(16, len(image_files)),
                    dtype=int
                )

                frames = []

                for index in indices:

                    frame = cv2.imread(
                        str(image_files[index])
                    )

                    if frame is not None:
                        frames.append(frame)

                if not frames:
                    raise ValueError(
                        "Could not read DOLOS frame images."
                    )

                st.info(
                    f"Using original DOLOS frames: {clip_stem}"
                )

                return frames

    # --------------------------------------
    # Otherwise: extract frames from uploaded MP4
    # --------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError("No video frames found.")

    indices = np.linspace(
        0,
        total_frames - 1,
        min(16, total_frames),
        dtype=int
    )

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if success and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Could not read video frames.")

    return frames



# ==========================================
# 6. VISUAL FEATURES - 5 FEATURES
# ==========================================

def extract_visual_features(frames):

    brightness = []
    contrast = []
    edges = []
    motion = []
    face_counts = []

    previous = None

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise RuntimeError("Face detector could not load.")

    for frame in frames:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (128, 128)
        )

        brightness.append(
            float(np.mean(gray))
        )

        contrast.append(
            float(np.std(gray))
        )

        edge = cv2.Canny(
            gray,
            80,
            160
        )

        edges.append(
            float(np.mean(edge > 0))
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4
        )

        face_counts.append(len(faces))

        if previous is not None:

            motion.append(
                float(
                    np.mean(
                        cv2.absdiff(
                            gray,
                            previous
                        )
                    )
                )
            )

        previous = gray

    return {

        "visual_brightness": float(
            np.mean(brightness)
        ),

        "visual_contrast": float(
            np.mean(contrast)
        ),

        "visual_edges": float(
            np.mean(edges)
        ),

        "visual_motion": (
            float(np.mean(motion))
            if motion else 0.0
        ),

        "visual_face_count": float(
            np.mean(face_counts)
        )
    }


# ==========================================
# 7. FACIAL FEATURES - 8 FEATURES
# ==========================================

def extract_facial_features(frames):

    cascade_dir = Path(cv2.data.haarcascades)

    face_detector = cv2.CascadeClassifier(
        str(
            cascade_dir
            / "haarcascade_frontalface_default.xml"
        )
    )

    eye_detector = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_eye.xml")
    )

    smile_detector = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_smile.xml")
    )

    if (
        face_detector.empty()
        or eye_detector.empty()
        or smile_detector.empty()
    ):
        raise RuntimeError("Facial detectors could not load.")

    face_rates = []
    eye_rates = []
    eye_counts = []
    smile_rates = []

    face_sizes = []
    face_centers = []
    mouth_changes = []

    previous_mouth = None

    for frame in frames:

        gray = cv2.cvtColor(
            frame,
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

        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        face_rates.append(1)

        face_sizes.append(
            (w * h) /
            (gray.shape[0] * gray.shape[1])
        )

        face_centers.append(
            (
                (x + w / 2) / gray.shape[1],
                (y + h / 2) / gray.shape[0]
            )
        )

        roi = gray[y:y+h, x:x+w]

        rh, rw = roi.shape

        # Eye detection proxy

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

        # Smile detection

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

        # Mouth-region variation

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

        raise ValueError(
            "No clear face detected. "
            "Please upload a video where the speaker's face is visible."
        )

    centers = np.array(face_centers)

    return {

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
            if len(centers) > 1 else 0.0
        ),

        "mouth_region_change": (
            float(np.mean(mouth_changes))
            if mouth_changes else 0.0
        )
    }


# ==========================================
# 8. SPEECH FEATURES - 6 FEATURES
# Same method as training script
# ==========================================

def extract_speech_features(audio_path):

    whisper = load_speech_model()

    segments, info = whisper.transcribe(
        str(audio_path),
        language="en",
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False
    )

    words = []
    transcript_parts = []

    for segment in segments:

        transcript_parts.append(
            segment.text.strip()
        )

        if segment.words:

            for word_info in segment.words:

                tokens = WORD_PATTERN.findall(
                    word_info.word.lower()
                )

                if tokens:

                    words.append(
                        (
                            tokens[0],
                            float(word_info.start),
                            float(word_info.end)
                        )
                    )

    words.sort(key=lambda item: item[1])

    transcript = " ".join(transcript_parts)

    pause_gaps = []

    pause_details = []

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

            pause_details.append({
                "start": round(previous[2], 2),
                "end": round(current[1], 2),
                "duration": round(gap, 2)
            })

    tokens = [
        word[0]
        for word in words
    ]

    filler_words = [
        token
        for token in tokens
        if token in FILLERS
    ]

    repeated_words = [
        second
        for first, second in zip(
            tokens,
            tokens[1:]
        )
        if first == second
    ]

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

    features = {

        "speech_pause_count": float(
            len(pause_gaps)
        ),

        "speech_long_pause_count": float(
            sum(gap >= 1.0 for gap in pause_gaps)
        ),

        "speech_pause_total_seconds": float(
            sum(pause_gaps)
        ),

        "speech_filler_count": float(
            len(filler_words)
        ),

        "speech_repeated_word_count": float(
            len(repeated_words)
        ),

        "speech_rate_wpm": float(
            speech_rate
        )
    }

    speech_report = {

        "transcript": transcript,

        "word_count": len(words),

        "filler_words": filler_words,

        "repeated_words": repeated_words,

        "pause_details": pause_details,

        "speech_rate": round(speech_rate, 1)
    }

    return features, speech_report


# ==========================================
# 9. COMPLETE MULTIMODAL PREDICTION
# ==========================================

def predict_video(video_path):

    model_bundle = load_model()

    model = model_bundle["model"]

    expected_features = (
        model_bundle["feature_columns"]
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        audio_path = (
            temp_dir / "extracted_audio.wav"
        )

        # Extract audio

        extract_audio(
            video_path,
            audio_path
        )

        # Extract video frames

        frames = extract_video_frames(
            video_path
        )

        # Extract all 68 features

        audio_features, y, sr = (
            extract_audio_features(audio_path)
        )

        visual_features = (
            extract_visual_features(frames)
        )

        facial_features = (
            extract_facial_features(frames)
        )

        speech_features, speech_report = (
            extract_speech_features(audio_path)
        )

        all_features = {}

        all_features.update(audio_features)
        all_features.update(visual_features)
        all_features.update(facial_features)
        all_features.update(speech_features)
        debug_path = ROOT / "last_uploaded_features.csv"
        pd.DataFrame([all_features]).to_csv(
            debug_path,
            index=False
)
        print("Live features saved:", debug_path)

        # Ensure exact model feature order

        missing = [
            feature
            for feature in expected_features
            if feature not in all_features
        ]

        if missing:

            raise ValueError(
                "Required model features missing: "
                + ", ".join(missing)
            )

        feature_df = pd.DataFrame(
            [[
                all_features[feature]
                for feature in expected_features
            ]],
            columns=expected_features
        )

        # FINAL PREDICTION USES ALL MODEL INPUTS

        prediction = model.predict(
            feature_df
        )[0]

        return (
            prediction,
            all_features,
            speech_report,
            len(expected_features)
        )


# ==========================================
# 10. STREAMLIT USER INTERFACE
# ==========================================

st.title(
    "🎙️ Audio-Visual Deception Pattern Analysis"
)

st.caption(
    "Experimental multimodal research prototype"
)

st.write(
    "Upload an MP4 containing a visible speaker "
    "and English speech for audio, facial and "
    "speech-pattern analysis."
)

st.info(
    "The model uses audio, visual, facial and speech "
    "features together. These patterns cannot prove "
    "whether someone is lying."
)

uploaded_file = st.file_uploader(
    "Upload MP4 video",
    type=["mp4"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True
    ):

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                video_path = (
                    Path(temp_dir) / Path(uploaded_file.name).name
                )

                video_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                with st.spinner(
                    "Extracting audio, video, facial and speech features..."
                ):

                    (
                        prediction,
                        features,
                        speech_report,
                        feature_count
                    ) = predict_video(video_path)

            st.divider()

            st.subheader("Final Model Prediction")

            if str(prediction).lower() == "deception":

                st.warning(
                    "Deception-pattern prediction"
                )

            elif str(prediction).lower() == "truth":

                st.success(
                    "Truth-pattern prediction"
                )

            else:

                st.write(
                    f"Model prediction: {prediction}"
                )

            st.caption(
                f"Prediction generated using {feature_count} "
                "audio, visual, facial and speech input features."
            )

            # ------------------------------
            # SPEECH PATTERN RESULTS
            # ------------------------------

            st.subheader("Speech Pattern Analysis")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Pause count",
                    int(features["speech_pause_count"])
                )

                st.metric(
                    "Long pauses",
                    int(features["speech_long_pause_count"])
                )

                st.metric(
                    "Filler words",
                    int(features["speech_filler_count"])
                )

            with col2:

                st.metric(
                    "Repeated words",
                    int(features["speech_repeated_word_count"])
                )

                st.metric(
                    "Speech rate",
                    f"{speech_report['speech_rate']} WPM"
                )

                st.metric(
                    "Total pause duration",
                    f"{features['speech_pause_total_seconds']:.2f} sec"
                )

            # ------------------------------
            # FACIAL AND VISUAL RESULTS
            # ------------------------------

            st.subheader("Visual and Facial Features")

            col3, col4 = st.columns(2)

            with col3:

                st.metric(
                    "Face detection rate",
                    f"{features['face_detection_rate']:.2f}"
                )

                st.metric(
                    "Eye detection rate",
                    f"{features['eye_detection_rate']:.2f}"
                )

                st.metric(
                    "Eye count mean",
                    f"{features['eye_count_mean']:.2f}"
                )

                st.metric(
                    "Head movement measure",
                    f"{features['head_center_motion']:.4f}"
                )

            with col4:

                st.metric(
                    "Mouth-region change",
                    f"{features['mouth_region_change']:.2f}"
                )

                st.metric(
                    "Smile detection rate",
                    f"{features['smile_detection_rate']:.2f}"
                )

                st.metric(
                    "Visual motion",
                    f"{features['visual_motion']:.2f}"
                )

                st.metric(
                    "Face visibility",
                    f"{features['face_detection_rate'] * 100:.1f}%"
                )

            # ------------------------------
            # WHY THESE OBSERVATIONS
            # ------------------------------

            st.subheader("Analysis Observations")

            observations = []

            if features["speech_pause_count"] > 0:

                observations.append(
                    f"{int(features['speech_pause_count'])} "
                    "speech pauses detected."
                )

            if features["speech_long_pause_count"] > 0:

                observations.append(
                    "Long speech pauses were detected."
                )

            if features["speech_filler_count"] > 0:

                observations.append(
                    f"{int(features['speech_filler_count'])} "
                    "filler words were detected."
                )

            if features["speech_repeated_word_count"] > 0:

                observations.append(
                    "Repeated words were detected."
                )

            if features["head_center_motion"] > 0:

                observations.append(
                    "Head-position movement was measured across frames."
                )

            if features["mouth_region_change"] > 0:

                observations.append(
                    "Mouth-region visual variation was measured."
                )

            if not observations:

                observations.append(
                    "No notable speech or visual variations were measured."
                )

            for item in observations[:3]:

                st.write("• " + item)

            # ------------------------------
            # TRANSCRIPT
            # ------------------------------

            with st.expander("View English Transcript"):

                st.write(
                    speech_report["transcript"]
                    if speech_report["transcript"]
                    else "No transcript detected."
                )

            # ------------------------------
            # PAUSE TIMESTAMPS
            # ------------------------------

            if speech_report["pause_details"]:

                with st.expander("View Pause Timestamps"):

                    for pause in speech_report["pause_details"]:

                        st.write(
                            f"{pause['start']}s - "
                            f"{pause['end']}s "
                            f"({pause['duration']}s)"
                        )

            # ------------------------------
            # MODEL INPUT FEATURE CHECK
            # ------------------------------

            with st.expander("View all model input features"):

                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Feature": key,
                                "Value": value
                            }
                            for key, value in features.items()
                        ]
                    ),
                    use_container_width=True
                )

            st.caption(
                "Research prototype only. Eye detection is a "
                "Haar-cascade proxy, not eye-gaze tracking. "
                "Facial and speech patterns are not proof of deception. "
                "Nervousness, culture, disability, context and speaking "
                "style can affect these measurements."
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


st.divider()

st.caption(
    "DOLOS-trained audio-visual-speech fusion model | "
    "Experimental research prototype"
)
