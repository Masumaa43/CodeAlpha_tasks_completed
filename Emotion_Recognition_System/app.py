import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.predict_emotion import predict_emotion


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VocalSense | Speech Emotion Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #392078, #6339B8, #35166D);
}

/* Hide Streamlit default elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main dashboard container */

.block-container {
    max-width: 1450px;
    padding: 2rem 2.5rem;
    background: #F5F3FA;
    border-radius: 30px;
    margin-top: 35px;
    margin-bottom: 35px;
}

/* Header */

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px 25px 10px;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #29213B;
}

.brand-icon {
    background: #633BC1;
    color: white;
    padding: 11px 14px;
    border-radius: 50%;
    margin-right: 12px;
}

/* Main heading */

.page-heading {
    color: #302443;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 4px;
}

.page-subheading {
    color: #9389A5;
    font-size: 13px;
    margin-bottom: 25px;
}

/* Metric cards */

.metric-card {
    background: white;
    padding: 19px;
    border-radius: 20px;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(70, 45, 110, 0.05);
    border: 1px solid #F0ECF7;
}

.metric-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    margin-bottom: 12px;
}

.metric-title {
    color: #9B91A9;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #302443;
    font-size: 23px;
    font-weight: 800;
    margin-top: 5px;
}

/* Content cards */

.content-card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 5px 18px rgba(70, 45, 110, 0.05);
    border: 1px solid #F0ECF7;
    margin-bottom: 18px;
}

.card-heading {
    color: #302443;
    font-size: 19px;
    font-weight: 750;
}

.card-description {
    color: #A39AAF;
    font-size: 12px;
    margin-top: 4px;
}

/* Prediction result */

.prediction-box {
    background: linear-gradient(135deg, #633BC1, #9672E5);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 20px rgba(99, 59, 193, 0.18);
}

.prediction-label {
    font-size: 13px;
    opacity: 0.85;
}

.prediction-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background: #F8F6FC;
    border: 1px dashed #B9A3E5;
    padding: 15px;
    border-radius: 15px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(135deg, #633BC1, #8C62DC);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #43218F;
    color: white;
    transform: translateY(-2px);
}

/* Download button */

.stDownloadButton > button {
    background: #F0EAFB;
    color: #633BC1;
    border: 1px solid #DCCDF5;
    border-radius: 10px;
    font-weight: 600;
}

/* Sidebar-like navigation rail */

.nav-rail {
    background: linear-gradient(180deg, #633BC1, #49218F);
    border-radius: 20px;
    padding: 18px 8px;
    text-align: center;
    color: white;
    font-size: 22px;
    min-height: 360px;
}

/* Footer */

.footer {
    text-align: center;
    color: #A49AB2;
    font-size: 11px;
    padding: 20px;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# COLORS
# --------------------------------------------------

emotion_colors = {
    "happy": "#43A982",
    "sad": "#729BE8",
    "angry": "#E47B86"
}

emotion_icons = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠"
}


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

if "audio_name" not in st.session_state:
    st.session_state.audio_name = None

if "duration" not in st.session_state:
    st.session_state.duration = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="top-header">
    <div class="brand">
        <span class="brand-icon">🎙</span>
        VocalSense
    </div>
    <div style="color:#81758F; font-size:13px;">
        Speech Emotion Analytics &nbsp; | &nbsp; AI Dashboard
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown("""
<div class="page-heading">Speech Emotion Dashboard</div>
<div class="page-subheading">
Analyze voice recordings and understand the emotions behind speech.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# AUDIO UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your voice recording",
    type=["wav", "mp3", "m4a"],
    help="Upload an audio recording to analyze its emotional tone."
)


# --------------------------------------------------
# AUDIO PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    if (
        st.session_state.audio_name != uploaded_file.name
        or st.session_state.audio_data is None
    ):

        st.session_state.prediction = None
        st.session_state.scores = None

        st.session_state.audio_name = uploaded_file.name
        st.session_state.audio_data = uploaded_file.getvalue()

        try:
            audio, sr = librosa.load(
                uploaded_file,
                sr=None,
                mono=True
            )

            st.session_state.duration = len(audio) / sr

        except Exception as e:
            st.error(f"Unable to read audio file: {e}")
            st.stop()


# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------

if st.session_state.audio_data is not None:

    duration_text = (
        f"{st.session_state.duration:.2f} sec"
        if st.session_state.duration is not None
        else "N/A"
    )

    audio_status = "Ready"
    prediction_status = (
        "Completed"
        if st.session_state.prediction
        else "Pending"
    )

else:
    duration_text = "0 sec"
    audio_status = "Waiting"
    prediction_status = "Pending"


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="background:#EEE5FF;color:#7548C9;">
            🎙️
        </div>
        <div class="metric-title">Audio Status</div>
        <div class="metric-value">{audio_status}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="background:#E5F4ED;color:#43A982;">
            ⏱️
        </div>
        <div class="metric-title">Audio Duration</div>
        <div class="metric-value">{duration_text}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="background:#E9EDFC;color:#729BE8;">
            🧠
        </div>
        <div class="metric-title">Model Status</div>
        <div class="metric-value">Ready</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="background:#FCE9ED;color:#E47B86;">
            📊
        </div>
        <div class="metric-title">Analysis Status</div>
        <div class="metric-value">{prediction_status}</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# --------------------------------------------------
# MAIN ANALYSIS SECTION
# --------------------------------------------------

left, right = st.columns([2.2, 1])

with left:

    st.markdown("""
    <div class="content-card">
        <div class="card-heading">Voice Signal Analysis</div>
        <div class="card-description">
            Visual representation of the uploaded speech recording
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.audio_data is not None:

        audio_bytes = st.session_state.audio_data

        st.audio(audio_bytes)

        try:
            audio, sr = librosa.load(
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(st.session_state.audio_name).suffix
                ).name,
                sr=None
            ) if False else librosa.load(
                __import__("io").BytesIO(audio_bytes),
                sr=None,
                mono=True
            )

            time_axis = np.arange(len(audio)) / sr

            # Limit points for a smooth and responsive chart
            step = max(1, len(audio) // 3000)

            waveform = go.Figure()

            waveform.add_trace(
                go.Scatter(
                    x=time_axis[::step],
                    y=audio[::step],
                    mode="lines",
                    line=dict(
                        color="#8154D6",
                        width=1.5
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(129,84,214,0.13)",
                    name="Audio Waveform"
                )
            )

            waveform.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                font=dict(color="#81758F"),
                showlegend=False,
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#F0ECF7"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#F0ECF7"
                )
            )

            st.plotly_chart(
                waveform,
                use_container_width=True
            )

        except Exception as e:
            st.warning(f"Could not generate waveform: {e}")

    else:
        st.info("Upload an audio recording to view its waveform.")


with right:

    st.markdown("""
    <div class="content-card">
        <div class="card-heading">Run Analysis</div>
        <div class="card-description">
            Start AI-powered emotion detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:

        if st.button("Analyze Voice 🎯"):

            temp_path = None

            try:
                suffix = Path(uploaded_file.name).suffix or ".wav"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(st.session_state.audio_data)
                    temp_path = temp_file.name

                with st.spinner("Analyzing voice emotions..."):

                    emotion, scores = predict_emotion(temp_path)

                    st.session_state.prediction = str(emotion).lower()
                    st.session_state.scores = scores

                st.success("Analysis completed successfully!")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.info("Upload an audio file first.")

    st.markdown("""
    <div class="content-card">
        <div class="card-heading">Supported Emotions</div>
        <div class="card-description">
            The model recognizes three emotional categories.
        </div>
        <br>
        <div style="font-size:14px;line-height:2.5;color:#625875;">
            😊 &nbsp; Happy<br>
            😢 &nbsp; Sad<br>
            😠 &nbsp; Angry
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

if st.session_state.prediction is not None:

    emotion = st.session_state.prediction
    scores = st.session_state.scores

    confidence = float(scores.get(emotion, 0))

    # Handle scores stored as percentages
    if confidence > 1:
        confidence = confidence / 100

    confidence = max(0, min(confidence, 1))

    st.write("")

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">DETECTED EMOTION</div>
        <div class="prediction-value">
            {emotion_icons.get(emotion, "🎙️")}
            {emotion.capitalize()}
        </div>
        <div style="font-size:14px;margin-top:8px;">
            Confidence Score: {confidence * 100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# LOWER ANALYTICS SECTION
# --------------------------------------------------

st.write("")

col1, col2 = st.columns(2)


# --------------------------------------------------
# EMOTION CONFIDENCE CHART
# --------------------------------------------------

with col1:

    st.markdown("""
    <div class="content-card">
        <div class="card-heading">Emotion Confidence</div>
        <div class="card-description">
            Confidence distribution across emotion categories
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.scores is not None:

        scores = st.session_state.scores

        labels = list(scores.keys())
        values = []

        for value in scores.values():
            value = float(value)

            if value > 1:
                value = value / 100

            values.append(value * 100)

        bar_colors = [
            emotion_colors.get(str(label).lower(), "#9673D2")
            for label in labels
        ]

        bar_chart = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=bar_colors,
                    text=[f"{v:.1f}%" for v in values],
                    textposition="outside"
                )
            ]
        )

        bar_chart.update_layout(
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Confidence (%)",
            font=dict(color="#81758F"),
            yaxis=dict(
                range=[0, 110],
                gridcolor="#F0ECF7"
            ),
            xaxis=dict(
                showgrid=False
            ),
            showlegend=False
        )

        st.plotly_chart(
            bar_chart,
            use_container_width=True
        )

    else:
        st.info("Emotion confidence will appear after analysis.")


# --------------------------------------------------
# MEL SPECTROGRAM
# --------------------------------------------------

with col2:

    st.markdown("""
    <div class="content-card">
        <div class="card-heading">Mel Spectrogram</div>
        <div class="card-description">
            Frequency representation of the voice signal
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.audio_data is not None:

        try:
            import io

            audio, sr = librosa.load(
                io.BytesIO(st.session_state.audio_data),
                sr=None,
                mono=True
            )

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=64
            )

            mel_db = librosa.power_to_db(
                mel_spec,
                ref=np.max
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=mel_db,
                    colorscale=[
                        [0, "#F0EAFB"],
                        [0.3, "#C8B1F0"],
                        [0.6, "#9673D2"],
                        [1, "#49218F"]
                    ],
                    colorbar=dict(
                        title="dB"
                    )
                )
            )

            fig.update_layout(
                height=300,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Time Frames",
                yaxis_title="Mel Frequency",
                font=dict(color="#81758F")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:
            st.warning(f"Could not generate spectrogram: {e}")

    else:
        st.info("Upload an audio file to view its Mel spectrogram.")


# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

if st.session_state.prediction is not None:

    st.write("")

    emotion = st.session_state.prediction
    scores = st.session_state.scores

    report = f"""
VOCALSENSE - SPEECH EMOTION ANALYSIS REPORT

Audio File: {st.session_state.audio_name}

Audio Duration: {st.session_state.duration:.2f} seconds

Detected Emotion: {emotion.capitalize()}

Emotion Confidence Scores:
"""

    for label, value in scores.items():

        value = float(value)

        if value > 1:
            value = value / 100

        report += f"\n{str(label).capitalize()}: {value * 100:.2f}%"

    report += """

Model: Speech Emotion Recognition Model
Recognized Categories: Happy, Sad, Angry

Developed by: Masuma Akhtar
"""

    st.download_button(
        label="⬇ Download Analysis Report",
        data=report,
        file_name="VocalSense_Emotion_Report.txt",
        mime="text/plain"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    VocalSense | Speech Emotion Recognition System
    <br>
    Developed by Masuma Akhtar | AI & Machine Learning
</div>
""", unsafe_allow_html=True)