import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
OUTPUT_DIR = BASE_DIR / "outputs"


def evaluate():

    print("Loading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    x_test = np.load(
        DATA_DIR / "test_images.npy",
        mmap_mode="r"
    )

    y_test = np.load(
        DATA_DIR / "test_labels.npy",
        mmap_mode="r"
    )

    print("Generating predictions...")

    predictions = model.predict(
        tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)
        ).batch(128),
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predicted_labels,
            zero_division=0
        )
    )

    accuracy = np.mean(
        predicted_labels == y_test
    )

    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predicted_labels
    )

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False
    )

    plt.title("EMNIST ByClass Confusion Matrix")

    plt.xlabel("Predicted Character")

    plt.ylabel("Actual Character")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plt.close()

    print("\nEvaluation completed!")


if __name__ == "__main__":
    evaluate()