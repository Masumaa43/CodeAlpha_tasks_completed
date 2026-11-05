import os
import joblib
import pandas as pd


def predict_loan_status():
    # Step 1: Check for the trained model
    model_path = "models/credit_scoring_model.pkl"

    if not os.path.exists(model_path):
        print("Trained model not found.")
        print("Run: python -m src.train_model")
        return

    # Step 2: Load the model
    model = joblib.load(model_path)

    print("\n========== CREDIT SCORING SYSTEM ==========")

    try:
        # Step 3: Collect borrower information
        person_age = int(input("Enter age: "))
        person_income = float(input("Enter annual income: "))

        person_home_ownership = input(
            "Home ownership (RENT/OWN/MORTGAGE/OTHER): "
        ).strip().upper()

        person_emp_length = float(
            input("Employment length in years: ")
        )

        loan_intent = input(
            "Loan intent: "
        ).strip().upper()

        loan_grade = input(
            "Loan grade (A-G): "
        ).strip().upper()

        loan_amnt = float(input("Loan amount: "))
        loan_int_rate = float(input("Interest rate: "))

        loan_percent_income = float(
            input("Loan-to-income ratio: ")
        )

        cb_person_default_on_file = input(
            "Previous default on file (Y/N): "
        ).strip().upper()

        cb_person_cred_hist_length = int(
            input("Credit history length in years: ")
        )

        # Step 4: Create the input DataFrame
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

        # Step 5: Make the prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Find the probability associated with class 1
        classes = list(model.classes_)
        default_index = classes.index(1)
        default_probability = probabilities[default_index]

        # Step 6: Display the result
        print("\n========== PREDICTION RESULT ==========")

        if prediction == 1:
            print("Model prediction: Default class")
        else:
            print("Model prediction: Non-default class")

        print(
            "Estimated model default probability: "
            f"{default_probability * 100:.2f}%"
        )

        print(
            "\nThis is an educational model prediction, "
            "not a real lending decision."
        )

    except (ValueError, KeyError) as error:
        print("Invalid input:", error)


if __name__ == "__main__":
    predict_loan_status()