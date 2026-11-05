import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from src.data_preprocessing import load_data, split_data


def evaluate_model():
    # Step 1: Load the dataset
    df = load_data()

    # Step 2: Separate features and target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Step 3: Create the test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Load the saved model
    model_path = "models/credit_scoring_model.pkl"

    if not os.path.exists(model_path):
        print("Trained model not found.")
        print("Run: python -m src.train_model")
        return

    model = joblib.load(model_path)

    # Step 5: Generate predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Step 6: Calculate evaluation metrics
    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_pred, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_test, y_probability):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Step 7: Create the results folder
    os.makedirs("results", exist_ok=True)

    # Step 8: Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Credit Scoring Model - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Step 9: Plot the ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        y_probability
    )

    plt.title("Credit Scoring Model - ROC Curve")
    plt.tight_layout()

    plt.savefig("results/roc_curve.png")
    plt.close()

    print("\nEvaluation completed successfully!")
    print("Confusion matrix: results/confusion_matrix.png")
    print("ROC curve: results/roc_curve.png")


if __name__ == "__main__":
    evaluate_model()