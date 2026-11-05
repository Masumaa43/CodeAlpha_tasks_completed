import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
from PIL import Image

import sys

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from src.predict import predict_character


st.set_page_config(
    page_title="Handwriting Recognition",
    page_icon="✍️",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

.title {
    text-align: center;
    color: #173b70;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
}

.result {
    background-color: #e8f1ff;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="title">Handwritten Character Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered character recognition using CNN and Deep Learning</div>',
    unsafe_allow_html=True
)

st.divider()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Home",
        "Character Recognition",
        "Model Performance",
        "About Project"
    ]
)


# Home Page
if page == "Home":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset",
        "EMNIST"
    )

    col2.metric(
        "Classes",
        "62"
    )

    col3.metric(
        "Model",
        "CNN"
    )

    st.subheader("Project Objective")

    st.write(
        """
        This project uses deep learning and image processing
        to identify handwritten digits and English alphabets.

        The system is trained using the EMNIST ByClass dataset
        and a Convolutional Neural Network.
        """
    )

    st.subheader("Recognized Characters")

    st.write(
        "Digits: 0–9"
    )

    st.write(
        "Uppercase: A–Z"
    )

    st.write(
        "Lowercase: a–z"
    )


# Recognition Page
elif page == "Character Recognition":

    st.header("Upload Handwritten Character")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Prediction")

            if st.button(
                "Recognize Character",
                type="primary"
            ):

                try:

                    character, confidence, probabilities = (
                        predict_character(image_array)
                    )

                    st.success(
                        f"Predicted Character: {character}"
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    # Top 5 predictions
                    top_indices = np.argsort(
                        probabilities
                    )[-5:][::-1]

                    st.subheader(
                        "Top 5 Predictions"
                    )

                    for index in top_indices:

                        label = (
                            list("0123456789")
                            + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                            + list("abcdefghijklmnopqrstuvwxyz")
                        )[index]

                        score = probabilities[index] * 100

                        st.write(
                            f"{label}: {score:.2f}%"
                        )

                        st.progress(
                            float(probabilities[index])
                        )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )


# Performance Page
elif page == "Model Performance":

    st.header("Model Performance")

    accuracy_path = BASE_DIR / "outputs" / "accuracy.png"

    loss_path = BASE_DIR / "outputs" / "loss.png"

    confusion_path = (
        BASE_DIR / "outputs" / "confusion_matrix.png"
    )

    if accuracy_path.exists():

        st.subheader("Accuracy Graph")

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    else:
        st.info("Train the model to generate accuracy graph.")

    if loss_path.exists():

        st.subheader("Loss Graph")

        st.image(
            str(loss_path),
            use_container_width=True
        )

    else:
        st.info("Loss graph is not available yet.")

    if confusion_path.exists():

        st.subheader("Confusion Matrix")

        st.image(
            str(confusion_path),
            use_container_width=True
        )


# About Page
elif page == "About Project":

    st.header("About the Project")

    st.write(
        """
        Handwritten Character Recognition is a deep learning
        project designed to identify handwritten digits and
        English alphabets.

        Technologies Used:

        - Python
        - TensorFlow
        - Keras
        - CNN
        - OpenCV
        - NumPy
        - Pandas
        - Streamlit

        Dataset: EMNIST ByClass

        Model: Convolutional Neural Network
        """
    )

    st.info(
        "Future Scope: CRNN and LSTM for handwritten "
        "word and sentence recognition."
    )


st.divider()

st.caption(
    "Handwritten Character Recognition System | "
    "Developed using Python and Deep Learning"
)