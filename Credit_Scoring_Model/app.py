
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CreditIQ | Credit Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# 2. PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F3F6FC;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101C35, #19345D);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #142644;
}

.sub-title {
    color: #718096;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #142644;
    margin-top: 20px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(20,38,68,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #718096;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #142644;
    font-weight: 800;
}

.stButton button,
.stFormSubmitButton button {
    background: #2457D6;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 45px;
    font-weight: 700;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #173EAA;
    color: white;
}

.footer {
    text-align: center;
    color: #8290A5;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATASET LOADING
# =====================================================

@st.cache_data
def load_dataset():

    possible_paths = [

        BASE_DIR / "dataset" / "credit_risk_dataset.csv",

        BASE_DIR / "dataset" / "credit_data.csv",

        BASE_DIR / "data" / "credit_risk_dataset.csv",

        BASE_DIR / "data" / "credit_data.csv",

        BASE_DIR / "credit_risk_dataset.csv",

        BASE_DIR / "credit_data.csv"

    ]

    for path in possible_paths:

        if path.exists():

            try:
                return pd.read_csv(path), str(
                    path.relative_to(BASE_DIR)
                )

            except Exception as error:
                st.error(f"Dataset reading error: {error}")
                return None, None

    excluded = {
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        "site-packages"
    }

    for path in BASE_DIR.rglob("*.csv"):

        if any(part in excluded for part in path.parts):
            continue

        if path.name.lower() in {
            "credit_risk_dataset.csv",
            "credit_data.csv"
        }:

            try:
                return pd.read_csv(path), str(
                    path.relative_to(BASE_DIR)
                )

            except Exception as error:
                st.error(f"Dataset reading error: {error}")
                return None, None

    return None, None


df, dataset_location = load_dataset()


# =====================================================
# 4. MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():

    possible_models = [

        BASE_DIR / "models" / "credit_scoring_model.pkl",

        BASE_DIR / "models" / "model.pkl",

        BASE_DIR / "credit_scoring_model.pkl",

        BASE_DIR / "model.pkl"

    ]

    for path in possible_models:

        if path.exists():

            try:
                return joblib.load(path), str(
                    path.relative_to(BASE_DIR)
                )

            except Exception as error:
                st.error(f"Model loading error: {error}")
                return None, None

    return None, None


model, model_location = load_model()


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def header(title, description):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sub-title">{description}</div>',
        unsafe_allow_html=True
    )


# Corrected metric function: native Streamlit component

def metric(label, value):

    st.metric(
        label=label,
        value=value,
        border=True
    )


def chart_style(fig):

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#142644",
        margin=dict(l=20, r=20, t=55, b=20),
        height=350
    )

    return fig


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 💳 CreditIQ")

    st.caption("CREDIT RISK INTELLIGENCE")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Risk Prediction",
            "Data Analytics",
            "Model Information"
        ]
    )

    st.markdown("---")

    st.markdown("### SYSTEM STATUS")

    if df is not None:
        st.success("Dataset Loaded")
    else:
        st.error("Dataset Not Found")

    if model is not None:
        st.success("ML Model Loaded")
    else:
        st.error("ML Model Not Found")

    st.markdown("---")

    st.caption("CodeAlpha Machine Learning Internship")

    st.caption("Developed by Masuma Akhtar")


# =====================================================
# 7. DASHBOARD
# =====================================================

if page == "Dashboard":

    header(
        "Credit Risk Dashboard",
        "Monitor loan applications and analyze credit default patterns."
    )

    if df is None:

        st.error("Dataset could not be found.")

        st.write("Expected dataset location:")

        st.code(
            "Credit Scoring Model/dataset/credit_risk_dataset.csv"
        )

    else:

        total = len(df)

        if "loan_status" in df.columns:

            defaults = int(
                (df["loan_status"] == 1).sum()
            )

            non_defaults = int(
                (df["loan_status"] == 0).sum()
            )

            default_rate = (
                defaults / total * 100 if total else 0
            )

        else:

            defaults = 0
            non_defaults = 0
            default_rate = 0

        st.markdown(
            '<div class="section-title">Portfolio Overview</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric("TOTAL APPLICANTS", f"{total:,}")

        with c2:
            metric("DEFAULT CASES", f"{defaults:,}")

        with c3:
            metric("NON-DEFAULT CASES", f"{non_defaults:,}")

        with c4:
            metric("DEFAULT RATE", f"{default_rate:.2f}%")

        st.markdown(
            '<div class="section-title">Credit Portfolio Analytics</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns(2)

        if "loan_status" in df.columns:

            with left:

                risk_data = pd.DataFrame({
                    "Category": [
                        "Non-Default",
                        "Default"
                    ],
                    "Applicants": [
                        non_defaults,
                        defaults
                    ]
                })

                fig = px.pie(
                    risk_data,
                    names="Category",
                    values="Applicants",
                    hole=0.65,
                    title="Loan Default Distribution"
                )

                st.plotly_chart(
                    chart_style(fig),
                    use_container_width=True
                )

        if "loan_grade" in df.columns:

            with right:

                grade_data = (
                    df["loan_grade"]
                    .value_counts()
                    .reset_index()
                )

                grade_data.columns = [
                    "Loan Grade",
                    "Applicants"
                ]

                fig = px.bar(
                    grade_data,
                    x="Loan Grade",
                    y="Applicants",
                    title="Loan Grade Distribution",
                    color="Applicants"
                )

                st.plotly_chart(
                    chart_style(fig),
                    use_container_width=True
                )

        st.markdown(
            '<div class="section-title">Applicant Records</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Dataset loaded from: {dataset_location}"
        )


# =====================================================
# 8. RISK PREDICTION
# =====================================================

elif page == "Risk Prediction":

    header(
        "Credit Risk Assessment",
        "Predict loan default risk using machine learning."
    )

    if model is None:

        st.error(
            "Trained model not found. Check your models folder."
        )

    else:

        st.markdown(
            '<div class="section-title">Borrower Information</div>',
            unsafe_allow_html=True
        )

        with st.form("prediction_form"):

            left, right = st.columns(2)

            with left:

                age = st.number_input(
                    "Applicant Age",
                    18, 100, 25
                )

                income = st.number_input(
                    "Annual Income (₹)",
                    min_value=1,
                    value=500000
                )

                home = st.selectbox(
                    "Home Ownership",
                    ["RENT", "OWN", "MORTGAGE", "OTHER"]
                )

                employment = st.number_input(
                    "Employment Length (Years)",
                    0.0, 60.0, 2.0
                )

                purpose = st.selectbox(
                    "Loan Purpose",
                    [
                        "PERSONAL",
                        "EDUCATION",
                        "MEDICAL",
                        "VENTURE",
                        "HOMEIMPROVEMENT",
                        "DEBTCONSOLIDATION"
                    ]
                )

                grade = st.selectbox(
                    "Loan Grade",
                    ["A", "B", "C", "D", "E", "F", "G"]
                )

            with right:

                amount = st.number_input(
                    "Loan Amount (₹)",
                    min_value=1000,
                    value=100000
                )

                interest = st.number_input(
                    "Interest Rate (%)",
                    0.0, 100.0, 10.0
                )

                default = st.selectbox(
                    "Previous Default",
                    ["N", "Y"]
                )

                history = st.number_input(
                    "Credit History Length (Years)",
                    0, 80, 3
                )

                ratio = amount / income

                st.metric(
                    "Loan-to-Income Ratio",
                    f"{ratio:.2f}"
                )

            submit = st.form_submit_button(
                "PREDICT CREDIT RISK",
                use_container_width=True
            )

        if submit:

            input_data = pd.DataFrame([{

                "person_age": age,

                "person_income": income,

                "person_home_ownership": home,

                "person_emp_length": employment,

                "loan_intent": purpose,

                "loan_grade": grade,

                "loan_amnt": amount,

                "loan_int_rate": interest,

                "loan_percent_income": ratio,

                "cb_person_default_on_file": default,

                "cb_person_cred_hist_length": history

            }])

            try:

                prediction = model.predict(input_data)[0]

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">Prediction Result</div>',
                    unsafe_allow_html=True
                )

                if prediction == 1:

                    st.error("HIGHER DEFAULT RISK")

                    st.write(
                        "The model predicts a higher likelihood of loan default."
                    )

                elif prediction == 0:

                    st.success("LOWER DEFAULT RISK")

                    st.write(
                        "The model predicts a lower likelihood of loan default."
                    )

                else:

                    st.warning(
                        f"Model prediction: {prediction}"
                    )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_data)[0]

                    classes = list(model.classes_)

                    if 1 in classes:

                        risk_probability = probabilities[
                            classes.index(1)
                        ]

                        st.metric(
                            "Estimated Default Probability",
                            f"{risk_probability * 100:.2f}%"
                        )

                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=risk_probability * 100,
                                number={"suffix": "%"},
                                title={
                                    "text": "Default Probability"
                                },
                                gauge={
                                    "axis": {
                                        "range": [0, 100]
                                    },
                                    "bar": {
                                        "color": "#2457D6"
                                    },
                                    "steps": [
                                        {
                                            "range": [0, 30],
                                            "color": "#D8F3E5"
                                        },
                                        {
                                            "range": [30, 60],
                                            "color": "#FFF0C2"
                                        },
                                        {
                                            "range": [60, 100],
                                            "color": "#FADBD8"
                                        }
                                    ]
                                }
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                st.info(
                    "Educational prediction only. This is not an actual loan approval."
                )

            except Exception as e:

                st.error(
                    "Prediction failed. Check the model's expected input features."
                )

                st.code(str(e))


# =====================================================
# 9. DATA ANALYTICS
# =====================================================

elif page == "Data Analytics":

    header(
        "Data Analytics",
        "Explore borrower information and discover dataset patterns."
    )

    if df is None:

        st.error("Dataset not found.")

    else:

        st.markdown("### Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.markdown("### Dataset Information")

        st.write("Total Rows:", len(df))

        st.write("Total Columns:", len(df.columns))

        st.write(
            "Missing Values:",
            int(df.isnull().sum().sum())
        )

        st.markdown("### Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected = st.selectbox(
                "Select a feature",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=selected,
                title=f"Distribution of {selected}",
                marginal="box"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        if len(numeric_columns) > 1:

            st.markdown("### Correlation Heatmap")

            correlation = df[numeric_columns].corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                title="Feature Correlation",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )


# =====================================================
# 10. MODEL INFORMATION
# =====================================================

elif page == "Model Information":

    header(
        "Model Information",
        "Technical overview of the Credit Scoring System."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### Project Details")

        st.write("**Project:** Credit Scoring Model")

        st.write("**Developer:** Masuma Akhtar")

        st.write("**Internship:** CodeAlpha")

        st.write("**Application:** Credit Default Prediction")

    with c2:

        st.markdown("### Technology Stack")

        st.write("- Python")

        st.write("- Pandas")

        st.write("- NumPy")

        st.write("- Scikit-learn")

        st.write("- Streamlit")

        st.write("- Plotly")

    st.markdown("---")

    st.markdown("### Model Status")

    if model is not None:

        st.success("Trained ML model loaded successfully.")

        st.write("Model Location:", model_location)

        st.write(
            "Model Type:",
            type(model).__name__
        )

        if hasattr(model, "classes_"):

            st.write(
                "Prediction Classes:",
                list(model.classes_)
            )

        if hasattr(model, "named_steps"):

            st.write(
                "Pipeline Steps:",
                list(model.named_steps.keys())
            )

    else:

        st.error("Model file not found.")

    st.markdown("---")

    st.markdown("### Dataset Status")

    if df is not None:

        st.success("Dataset loaded successfully.")

        st.write("Dataset Location:", dataset_location)

        st.write("Number of Records:", len(df))

        st.write("Number of Features:", len(df.columns))

    else:

        st.error("Dataset not loaded.")


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        CREDITIQ | CREDIT RISK ANALYTICS
        <br>
        Developed by Masuma Akhtar
        <br>
        CodeAlpha Machine Learning Internship
    </div>
    """,
    unsafe_allow_html=True
)
