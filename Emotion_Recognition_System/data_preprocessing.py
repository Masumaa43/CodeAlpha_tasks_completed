from pathlib import Path
import numpy as np
import librosa

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "audio_speech_actors_01-24_16k"
)

# RAVDESS Speech 16K settings
SAMPLE_RATE = 16000
DURATION = 3

SAMPLES_PER_AUDIO = SAMPLE_RATE * DURATION

# Emotion codes in RAVDESS
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_audio(file_path):

    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make all audio recordings 3 seconds long
    if len(audio) < SAMPLES_PER_AUDIO:

        audio = np.pad(
            audio,
            (0, SAMPLES_PER_AUDIO - len(audio))
        )

    else:

        audio = audio[:SAMPLES_PER_AUDIO]

    return audio


def get_emotion(file_path):

    filename = Path(file_path).stem

    parts = filename.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"Unexpected RAVDESS filename: {filename}"
        )

    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion code: {emotion_code}"
        )

    return EMOTION_MAP[emotion_code]


def get_actor_id(file_path):

    filename = Path(file_path).stem

    parts = filename.split("-")

    return int(parts[6])


def get_audio_files():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_DIR}"
        )

    files = sorted(
        DATASET_DIR.rglob("*.wav")
    )

    if not files:

        raise FileNotFoundError(
            "No WAV files found in the dataset folder."
        )

    return files