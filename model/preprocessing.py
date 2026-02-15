# model/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(filepath: str):
    df = pd.read_csv(filepath)  # comma separated
    return df


def preprocess_data(df):

    # Clean column names (good practice)
    df.columns = df.columns.str.strip()

    # Target column
    target_col = "deposit"

    X = df.drop(target_col, axis=1)
    y = df[target_col].map({"yes": 1, "no": 0})

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return X, y, preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )