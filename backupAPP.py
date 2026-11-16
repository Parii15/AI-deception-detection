
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import tempfile
import subprocess
import imageio_ffmpeg
import re

from pathlib import Path
from faster_whisper import WhisperModel


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "results" / "final_test_mfcc_v1"
    / "final_mfcc_svm.joblib"
)

st.set_page_config(
    page_title="Deception Pattern Detector",
    page_icon="🎙️",
    layout="centered"
)


# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_speech_model():
    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8"
    )


# =====================================
# EXTRACT AUDIO FROM MP4
# =====================================

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
            "Audio extraction failed: " + result.stderr[-1500:]
        )


# =====================================
# EXTRACT 39 MFCC FEATURES
# =====================================

def extract_mfcc_features(audio_path):

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

    features = {}

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

    return pd.DataFrame([features])


# =====================================
# PAUSE DETECTION
# =====================================

def analyze_pauses(audio_path):

    y, sr = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    duration = len(y) / sr

    if duration < 1:
        raise ValueError("Audio is too short.")

    intervals = librosa.effects.split(
        y,
        top_db=30,
        frame_length=1024,
        hop_length=256
    )

    pauses = []

    for i in range(len(intervals) - 1):

        end_current = intervals[i][1] / sr
        start_next = intervals[i + 1][0] / sr

        gap = start_next - end_current

        if gap >= 0.4:
            pauses.append({
                "start": round(end_current, 2),
                "end": round(start_next, 2),
                "duration": round(gap, 2)
            })

    long_pauses = [
        p for p in pauses
        if p["duration"] >= 1.0
    ]

    total_pause_time = sum(
        p["duration"] for p in pauses
    )

    speech_duration = sum(
        (end - start) / sr
        for start, end in intervals
    )

    return {
        "duration": round(duration, 2),
        "speech_duration": speech_duration,
        "pause_count": len(pauses),
        "long_pause_count": len(long_pauses),
        "total_pause_time": round(total_pause_time, 2),
        "pauses": pauses
    }


# =====================================
# HESITATION + SPEECH RATE ANALYSIS
# =====================================

def analyze_speech(audio_path, speech_duration):

    speech_model = load_speech_model()

    segments, info = speech_model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        vad_filter=True
    )

    words = []
    transcript_parts = []

    for segment in segments:

        transcript_parts.append(segment.text.strip())

        if segment.words:

            for word_info in segment.words:

                word = word_info.word.strip()

                if word:
                    words.append(word)

    transcript = " ".join(transcript_parts)

    # Clean words for analysis
    clean_words = [
        re.sub(r"[^a-zA-Z']", "", word).lower()
        for word in words
    ]

    clean_words = [
        word for word in clean_words if word
    ]

    # Filler word detection
    filler_words = {
        "um", "uh", "erm", "er", "hmm", "umm", "uhh"
    }

    detected_fillers = [
        word for word in clean_words
        if word in filler_words
    ]

    # Consecutive repeated word detection
    repeated_words = []

    for i in range(1, len(clean_words)):

        if clean_words[i] == clean_words[i - 1]:
            repeated_words.append(clean_words[i])

    # Speech rate calculation
    word_count = len(clean_words)

    if speech_duration > 0:
        speech_rate = (
            word_count / speech_duration
        ) * 60
    else:
        speech_rate = 0

    return {
        "transcript": transcript,
        "word_count": word_count,
        "speech_rate": round(speech_rate, 1),
        "filler_count": len(detected_fillers),
        "filler_words": detected_fillers,
        "repeated_count": len(repeated_words),
        "repeated_words": repeated_words
    }


# =====================================
# COMPLETE VIDEO ANALYSIS
# =====================================

def predict_video(video_path):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        audio_path = temp_dir / "extracted_audio.wav"

        extract_audio(video_path, audio_path)

        features = extract_mfcc_features(audio_path)

        pause_results = analyze_pauses(audio_path)

        speech_results = analyze_speech(
            audio_path,
            pause_results["speech_duration"]
        )

        model = load_model()

        prediction = model.predict(features)[0]

        return prediction, pause_results, speech_results


# =====================================
# STREAMLIT USER INTERFACE
# =====================================

st.title("🎙️ Deception Pattern Detector")

st.caption(
    "AI-based English speech audio analysis"
)

st.write(
    "Upload an MP4 video containing English speech "
    "to analyze its audio patterns."
)

st.info(
    "This is an experimental research prototype. "
    "Speech patterns cannot independently establish truth or deception."
)

uploaded_file = st.file_uploader(
    "Upload your MP4 video",
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

                video_path = Path(temp_dir) / "uploaded_video.mp4"

                video_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                with st.spinner(
                    "Analyzing audio, pauses and speech patterns..."
                ):

                    prediction, pause_results, speech_results = (
                        predict_video(video_path)
                    )

            st.divider()

            st.subheader("Analysis Result")

            # Existing SVM prediction
            if str(prediction).lower() == "deception":

                st.warning(
                    "Deception-pattern prediction"
                )

            elif str(prediction).lower() == "truth":

                st.success(
                    "Truth-pattern prediction"
                )

            else:

                st.write(f"Model prediction: {prediction}")

            # Pause analysis
            st.subheader("Speech Pattern Observations")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total pauses",
                    pause_results["pause_count"]
                )

            with col2:
                st.metric(
                    "Long pauses",
                    pause_results["long_pause_count"]
                )

            st.write(
                f"**Total pause duration:** "
                f"{pause_results['total_pause_time']} seconds"
            )

            # Speech rate
            st.metric(
                "Speech rate",
                f"{speech_results['speech_rate']} words/min"
            )

            # Hesitation analysis
            col3, col4 = st.columns(2)

            with col3:
                st.metric(
                    "Filler words",
                    speech_results["filler_count"]
                )

            with col4:
                st.metric(
                    "Repeated words",
                    speech_results["repeated_count"]
                )

            # Reasons
            observations = []

            if pause_results["long_pause_count"] > 0:
                observations.append(
                    "Long pauses detected in speech."
                )

            if pause_results["pause_count"] >= 5:
                observations.append(
                    "Frequent silent gaps detected."
                )

            if speech_results["filler_count"] > 0:
                observations.append(
                    "Speech hesitation markers detected."
                )

            if speech_results["repeated_count"] > 0:
                observations.append(
                    "Repeated words detected."
                )

            if speech_results["word_count"] > 0:
                observations.append(
                    f"Speech rate: "
                    f"{speech_results['speech_rate']} words per minute."
                )

            if not observations:
                observations.append(
                    "No notable pause or hesitation markers detected."
                )

            st.write("**Why these observations?**")

            for item in observations[:3]:
                st.write("• " + item)

            # Optional transcript
            with st.expander("View English transcript"):

                st.write(
                    speech_results["transcript"]
                    if speech_results["transcript"]
                    else "No speech transcript detected."
                )

            # Pause timestamps
            if pause_results["pauses"]:

                with st.expander("View pause timestamps"):

                    for pause in pause_results["pauses"]:

                        st.write(
                            f"{pause['start']}s - "
                            f"{pause['end']}s "
                            f"({pause['duration']}s)"
                        )

            st.caption(
                "SVM prediction is based on MFCC features. "
                "Pause, speech rate and hesitation observations "
                "are separate measurements. They do not prove "
                "deception. Normal thinking, nervousness and "
                "speaking style can also affect these patterns."
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


st.divider()

st.caption(
    "Research prototype | DOLOS-trained MFCC + SVM model"
)
