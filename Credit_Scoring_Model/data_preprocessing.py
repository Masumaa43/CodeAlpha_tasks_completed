import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_data():
    """Load the credit risk dataset."""

    file_path = "dataset/credit_risk_dataset.csv"

    df = pd.read_csv(file_path)

    if "loan_status" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'loan_status' column."
        )

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def prepare_data(df):
    """Separate features and target and prepare preprocessing."""

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return X, y, preprocessor


def split_data(X, y):
    """Split the dataset into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test