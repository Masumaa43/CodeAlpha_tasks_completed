import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.data_preprocessing import (
    load_data,
    prepare_data,
    split_data
)


def train_model():
    # Step 1: Load the dataset
    df = load_data()

    # Step 2: Prepare the data
    X, y, preprocessor = prepare_data(df)

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Create the model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Step 5: Train the model
    print("\nTraining the Logistic Regression model...")
    model.fit(X_train, y_train)

    # Step 6: Generate predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Step 7: Display evaluation metrics
    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(f"{roc_auc_score(y_test, y_probability):.4f}")

    # Step 8: Save the trained model
    os.makedirs("models", exist_ok=True)

    model_path = "models/credit_scoring_model.pkl"
    joblib.dump(model, model_path)

    print("\nModel training completed successfully!")
    print("Saved model:", model_path)


if __name__ == "__main__":
    train_model()