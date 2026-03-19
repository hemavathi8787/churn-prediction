import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.clean import load_data, clean_data, get_feature_types
from src.features import engineer_features


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes + scales numeric features
    - Imputes + ordinally encodes categorical features
    Fitted only on training data to prevent leakage.
    """
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_full_pipeline(classifier, num_cols: list, cat_cols: list) -> ImbPipeline:
    """
    Wrap preprocessor + SMOTE + classifier into one pipeline.
    Using imblearn Pipeline so SMOTE only runs during fit, not predict.
    """
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", classifier),
        ]
    )


def get_data_splits(data_path: str):
    """Load, clean, engineer, and split data. Returns train/test splits."""
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=["Churn", "TenureBucket"], errors="ignore")
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # preserve class ratio in both splits
    )

    num_cols, cat_cols = get_feature_types(df)
    # Remove TenureBucket from cat_cols (dropped above)
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    return X_train, X_test, y_train, y_test, num_cols, cat_cols
