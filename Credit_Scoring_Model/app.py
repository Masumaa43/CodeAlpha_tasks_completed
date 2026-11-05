import os
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_path = "models/credit_scoring_model.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    return None


st.title("Credit Scoring Model")
st.write(
    "An educational machine learning application "
    "for estimating loan-default risk."
)

model = load_model()

if model is None:
    st.error("Trained model not found.")

    st.info(
        "Run the following command in your VS Code terminal "
        "to train the model first:"
    )

    st.code("python -m src.train_model")

else:
    st.subheader("Enter Borrower Information")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25
        )

        person_income = st.number_input(
            "Annual Income",
            min_value=0,
            value=500000
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        person_emp_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=100.0,
            value=2.0
        )

        loan_intent = st.selectbox(
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

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

    with col2:
        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0,
            value=100000
        )

        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        loan_percent_income = st.number_input(
            "Loan-to-Income Ratio",
            min_value=0.0,
            max_value=10.0,
            value=0.20,
            step=0.01
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            ["N", "Y"]
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=100,
            value=3
        )

    if st.button("Predict Loan Risk"):
        input_data = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": (
                cb_person_default_on_file
            ),
            "cb_person_cred_hist_length": (
                cb_person_cred_hist_length
            )
        }])

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.warning(
                    "The model predicts the loan-default class."
                )
            else:
                st.success(
                    "The model predicts the non-default class."
                )

            st.metric(
                "Estimated Default Probability",
                f"{probability * 100:.2f}%"
            )

            st.caption(
                "Educational demonstration only. This prediction "
                "is not a guarantee or a real lending decision."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")