import os
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"

TRAIN_CSV = DATASET_DIR / "emnist-byclass-train.csv"
TEST_CSV = DATASET_DIR / "emnist-byclass-test.csv"

PROCESSED_DIR.mkdir(exist_ok=True)

# EMNIST ByClass sizes
TRAIN_SIZE = 697932
TEST_SIZE = 116323

IMAGE_SIZE = 28
PIXELS = 784
CHUNK_SIZE = 10000


def process_csv(csv_path, prefix, expected_size):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    image_path = PROCESSED_DIR / f"{prefix}_images.npy"
    label_path = PROCESSED_DIR / f"{prefix}_labels.npy"

    # Reuse processed files
    if image_path.exists() and label_path.exists():
        print(f"{prefix} data already processed.")
        return

    images = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_size, 28, 28, 1)
    )

    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int64,
        shape=(expected_size,)
    )

    start = 0

    print(f"\nProcessing {prefix} dataset...")

    for chunk in pd.read_csv(
        csv_path,
        header=None,
        chunksize=CHUNK_SIZE
    ):

        if chunk.shape[1] != 785:
            raise ValueError(
                "Expected 785 columns: 1 label + 784 pixels."
            )

        count = len(chunk)
        end = start + count

        if end > expected_size:
            raise ValueError("Unexpected number of CSV rows.")

        chunk_labels = chunk.iloc[:, 0].to_numpy(
            dtype=np.int64
        )

        chunk_images = chunk.iloc[:, 1:].to_numpy(
            dtype=np.uint8
        )

        # Convert pixels into 28x28 images
        chunk_images = chunk_images.reshape(
            -1, 28, 28
        )

        # Correct EMNIST CSV orientation
        chunk_images = np.transpose(
            chunk_images, (0, 2, 1)
        )

        images[start:end] = chunk_images[..., np.newaxis]
        labels[start:end] = chunk_labels

        start = end

        print(f"Processed {start}/{expected_size}")

    if start != expected_size:
        raise ValueError(
            f"Expected {expected_size} rows, found {start}."
        )

    images.flush()
    labels.flush()

    del images
    del labels

    print(f"{prefix} processing completed!")


def prepare_data():

    process_csv(
        TRAIN_CSV,
        "train",
        TRAIN_SIZE
    )

    process_csv(
        TEST_CSV,
        "test",
        TEST_SIZE
    )

    print("\nAll datasets prepared successfully!")


if __name__ == "__main__":
    prepare_data()