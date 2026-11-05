import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

# Character mapping
CHARACTERS = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)


def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Convert to black background and white strokes
    if gray.mean() > 127:
        gray = 255 - gray

    # Remove background noise
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find character boundaries
    coords = cv2.findNonZero(binary)

    if coords is None:
        raise ValueError("No handwritten character detected.")

    x, y, w, h = cv2.boundingRect(coords)

    character = binary[
        y:y+h,
        x:x+w
    ]

    # Resize while maintaining aspect ratio
    scale = 20 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        character,
        (new_w, new_h)
    )

    # Place character in 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    # Apply same orientation correction as training
    canvas = canvas.T

    canvas = canvas.reshape(
        1, 28, 28, 1
    )

    return canvas


def predict_character(image):

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    processed = preprocess_image(image)

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_character = CHARACTERS[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ] * 100

    return predicted_character, confidence, probabilities


if __name__ == "__main__":

    image_path = input(
        "Enter handwritten image path: "
    )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            "Unable to load image."
        )

    character, confidence, _ = predict_character(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    print("\nPredicted Character:", character)

    print("Confidence:", round(confidence, 2), "%")        

    