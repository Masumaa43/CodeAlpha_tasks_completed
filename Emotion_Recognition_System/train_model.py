import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt

from src.data_preprocessing import (
    get_audio_files,
    get_emotion,
    get_actor_id
)

from src.feature_extraction import extract_mfcc


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# -----------------------------------
# 1. Load dataset and extract features
# -----------------------------------

print("Loading audio dataset...")

audio_files = get_audio_files()

X = []
y = []
groups = []

for index, file_path in enumerate(audio_files):

    try:

        features = extract_mfcc(file_path)

        emotion = get_emotion(file_path)

        actor = get_actor_id(file_path)

        X.append(features)

        y.append(emotion)

        groups.append(actor)

    except Exception as error:

        print(
            f"Skipping {file_path.name}: {error}"
        )

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(audio_files)} files"
        )


X = np.array(X, dtype=np.float32)

y = np.array(y)

groups = np.array(groups)


print("\nDataset loaded successfully!")

print("Feature shape:", X.shape)

print("Number of samples:", len(X))

print("Number of emotions:", len(np.unique(y)))


# -----------------------------------
# 2. Encode emotion labels
# -----------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_.tolist()

print("\nEmotion classes:")

print(class_names)


# -----------------------------------
# 3. Split dataset by actor
# -----------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y_encoded, groups=groups)
)


X_train = X[train_indices]

X_test = X[test_indices]

y_train = y_encoded[train_indices]

y_test = y_encoded[test_indices]


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# -----------------------------------
# 4. Normalize MFCC features
# -----------------------------------

mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

std = np.maximum(std, 1e-8)


X_train = (X_train - mean) / std

X_test = (X_test - mean) / std


# -----------------------------------
# 5. Build CNN model
# -----------------------------------

number_of_classes = len(class_names)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2])
    ),

    tf.keras.layers.Conv1D(
        64,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.35),

    tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax"
    )

])


# -----------------------------------
# 6. Compile model
# -----------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


model.summary()


# -----------------------------------
# 7. Early stopping
# -----------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)


# -----------------------------------
# 8. Train model
# -----------------------------------

print("\nStarting CNN training...")

history = model.fit(

    X_train,

    y_train,

    validation_split=0.15,

    epochs=60,

    batch_size=32,

    callbacks=[early_stopping],

    verbose=1

)


# -----------------------------------
# 9. Evaluate model
# -----------------------------------

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


print("\nTest Accuracy:", test_accuracy)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predicted_classes,
        labels=np.arange(number_of_classes),
        target_names=class_names,
        zero_division=0
    )
)


# -----------------------------------
# 10. Confusion matrix
# -----------------------------------

cm = confusion_matrix(
    y_test,
    predicted_classes,
    labels=np.arange(number_of_classes)
)

plt.figure(figsize=(10, 7))

plt.imshow(cm)

plt.title("Emotion Recognition Confusion Matrix")

plt.xlabel("Predicted Emotion")

plt.ylabel("Actual Emotion")

plt.xticks(
    range(number_of_classes),
    class_names,
    rotation=45
)

plt.yticks(
    range(number_of_classes),
    class_names
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png"
)

plt.show()


# -----------------------------------
# 11. Training accuracy graph
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    MODEL_DIR / "accuracy_graph.png"
)

plt.show()


# -----------------------------------
# 12. Training loss graph
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.savefig(
    MODEL_DIR / "loss_graph.png"
)

plt.show()


# -----------------------------------
# 13. Save trained model
# -----------------------------------

model.save(
    MODEL_DIR / "emotion_model.keras"
)


# Save normalization values
np.savez(

    MODEL_DIR / "normalization.npz",

    mean=mean,

    std=std

)


# Save emotion labels
with open(
    MODEL_DIR / "labels.json",
    "w"
) as file:

    json.dump(
        class_names,
        file
    )


print("\nTraining completed successfully!")

print("Model saved inside models folder.")