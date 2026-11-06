import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 62


# Load processed dataset
def load_arrays():

    x_train = np.load(

        DATA_DIR / "train_images.npy",
        mmap_mode="r"
    )

    y_train = np.load(
        DATA_DIR / "train_labels.npy",
        mmap_mode="r"
    )

    x_test = np.load(
        DATA_DIR / "test_images.npy",
        mmap_mode="r"
    )

    y_test = np.load(
        DATA_DIR / "test_labels.npy",
        mmap_mode="r"
    )

    return x_train, y_train, x_test, y_test


# Create TensorFlow datasets
def create_dataset(images, labels, training=False):

    def generator():

        for i in range(len(labels)):
            yield images[i], np.int64(labels[i])

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(28, 28, 1),
                dtype=tf.uint8
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int64
            )
        )
    )

    if training:
        dataset = dataset.shuffle(
            20000,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# CNN Architecture
def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(28, 28, 1)
        ),

        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.RandomRotation(0.03),

        # First convolution block
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolution block
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third convolution block
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        # Flatten
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.4),

        # Output layer
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# Main training function
def train():

    print("Loading processed dataset...")

    x_train, y_train, x_test, y_test = load_arrays()

    print("Training images:", len(x_train))
    print("Testing images:", len(x_test))

    train_ds = create_dataset(
        x_train,
        y_train,
        training=True
    )

    test_ds = create_dataset(
        x_test,
        y_test
    )

    print("\nBuilding CNN model...")

    model = build_model()

    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )

    ]

    print("\nStarting CNN training...")

    history = model.fit(

        train_ds,

        validation_data=test_ds,

        epochs=EPOCHS,

        callbacks=callbacks

    )

    model.save(
        MODEL_DIR / "cnn_model.keras"
    )

    print("\nModel training completed!")

    # Accuracy graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("CNN Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(
        OUTPUT_DIR / "accuracy.png"
    )

    plt.close()

    # Loss graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("CNN Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(
        OUTPUT_DIR / "loss.png"
    )

    plt.close()

    print("Graphs saved in outputs folder.")


if __name__ == "__main__":
    train()