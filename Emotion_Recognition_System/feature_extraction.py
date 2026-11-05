import librosa
import numpy as np

from src.data_preprocessing import (
    load_audio,
    SAMPLE_RATE
)


def extract_mfcc(file_path, n_mfcc=40):

    audio = load_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=512
    )

    # Convert shape from (40, time) to (time, 40)
    mfcc = mfcc.T

    return mfcc.astype(np.float32)