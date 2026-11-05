import tempfile

import streamlit as st

import librosa
import librosa.display

import matplotlib.pyplot as plt

import numpy as np

from src.predict_emotion import predict_emotion


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(

    page_title="Emotion Recognition",

    page_icon="🎙️",

    layout="wide"

)


# -----------------------------------
# Custom styling
# -----------------------------------

st.markdown("""

<style>

.stApp {
    background-color: #f4f7fb;
}

h1 {
    color: #163a70;
}

h2, h3 {
    color: #234f87;
}

</style>

""", unsafe_allow_html=True)


# -----------------------------------
# Application title
# -----------------------------------

st.title("🎙️ Emotion Recognition from Speech")

st.write(

    "An AI-powered application that identifies "
    "human emotions from speech audio using "
    "MFCC feature extraction and a CNN model."

)


st.divider()


# -----------------------------------
# Upload audio
# -----------------------------------

st.subheader("Upload Your Audio")

uploaded_file = st.file_uploader(

    "Choose a WAV audio file",

    type=["wav"]

)


if uploaded_file is not None:

    st.success("Audio uploaded successfully!")

    # Play audio
    st.audio(
        uploaded_file,
        format="audio/wav"
    )


    # Save temporary audio file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        audio_path = temp_file.name


    # -----------------------------------
    # Audio visualization
    # -----------------------------------

    st.subheader("Audio Analysis")

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )


    col1, col2 = st.columns(2)


    # Waveform
    with col1:

        st.write("Waveform")

        fig, ax = plt.subplots(
            figsize=(8, 3)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax
        )

        ax.set_xlabel("Time")

        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

        plt.close(fig)


    # Mel spectrogram
    with col2:

        st.write("Mel Spectrogram")

        mel = librosa.feature.melspectrogram(

            y=audio,

            sr=sr,

            n_mels=64

        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        fig, ax = plt.subplots(
            figsize=(8, 3)
        )

        librosa.display.specshow(

            mel_db,

            sr=sr,

            x_axis="time",

            y_axis="mel",

            ax=ax

        )

        fig.colorbar(
            ax.collections[0],
            ax=ax,
            format="%+2.0f dB"
        )

        st.pyplot(fig)

        plt.close(fig)


    st.divider()


    # -----------------------------------
    # Emotion prediction
    # -----------------------------------

    st.subheader("Emotion Prediction")


    if st.button(
        "Analyze Emotion",
        type="primary"
    ):

        try:

            with st.spinner(
                "Analyzing speech..."
            ):

                emotion, scores = predict_emotion(
                    audio_path
                )


            st.success(
                f"Predicted Emotion: {emotion.upper()}"
            )


            st.subheader(
                "Emotion Confidence Scores"
            )


            for label, score in scores.items():

                st.write(
                    f"{label.capitalize()}: {score * 100:.2f}%"
                )

                st.progress(
                    float(score)
                )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


else:

    st.info(
        "Please upload a WAV audio file to begin."
    )


# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption(
    "Emotion Recognition from Speech | "
    "Built using Python, Librosa, TensorFlow and Streamlit"
)