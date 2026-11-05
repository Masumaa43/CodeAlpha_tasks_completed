import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.feature_extraction import extract_mfcc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"


def predict_emotion(audio_path):

    model_path = MODEL_DIR / "emotion_model.keras"

    normalization_path = MODEL_DIR / "normalization.npz"

    labels_path = MODEL_DIR / "labels.json"


    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Train the model first."
        )


    # Load trained model
    model = tf.keras.models.load_model(
        model_path
    )


    # Load normalization values
    normalization = np.load(
        normalization_path
    )

    mean = normalization["mean"]

    std = normalization["std"]


    # Load emotion labels
    with open(labels_path, "r") as file:

        class_names = json.load(file)


    # Extract MFCC features
    features = extract_mfcc(audio_path)


    # Normalize features
    features = (features - mean.reshape(1, -1)) / std.reshape(1, -1)


    # Add batch dimension
    features = np.expand_dims(
        features,
        axis=0
    )


    # Predict emotion
    probabilities = model.predict(
        features,
        verbose=0
    )[0]


    predicted_index = np.argmax(
        probabilities
    )

    predicted_emotion = class_names[
        predicted_index
    ]


    scores = {

        class_names[index]: float(
            probabilities[index]
        )

        for index in range(len(class_names))

    }


    return predicted_emotion, scores