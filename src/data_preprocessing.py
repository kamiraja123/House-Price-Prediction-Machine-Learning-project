"""
Data Preprocessing Module for House Price Prediction.

This module provides functions and utilities for:
- Loading the dataset
- Cleaning duplicates and initial data validation
- Checking for outliers
- Constructing scikit-learn Pipelines with ColumnTransformer
  (handling missing values, numerical scaling, and categorical encoding)
- Preventing data leakage during train/test splits
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define default feature columns
NUMERICAL_FEATURES: List[str] = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Parking",
    "YearBuilt",
]

CATEGORICAL_FEATURES: List[str] = [
    "Location",
    "PropertyType",
]

TARGET_COLUMN: str = "Price"


def get_default_data_path() -> str:
    """
    Dynamically locate the data/house_data.csv file relative to project root.

    Returns
    -------
    str
        Absolute path to the dataset CSV file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    return os.path.join(project_root, "data", "house_data.csv")


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the housing dataset from CSV file.

    Parameters
    ----------
    file_path : Optional[str]
        Path to CSV dataset. If None, uses default project data path.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if file_path is None:
        file_path = get_default_data_path()

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}. Please ensure house_data.csv exists."
        )

    df = pd.read_csv(file_path)
    return df


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identify and remove duplicate records from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned DataFrame and count of duplicate rows removed.
    """
    initial_count = len(df)
    df_cleaned = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df_cleaned)
    return df_cleaned, duplicates_removed


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Summarize count of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Missing values count per column.
    """
    return df.isnull().sum()


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect statistical outliers using the Interquartile Range (IQR) technique.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Numeric column to analyze for outliers.
    factor : float
        IQR multiplier (standard is 1.5).

    Returns
    -------
    pd.DataFrame
        Subset of rows classified as outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    col_data = df[column].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def create_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that cleanly handles:
    1. Numerical features:
       - Missing values: Median imputation (robust to skew)
       - Feature scaling: StandardScaler (zero mean, unit variance)
    2. Categorical features:
       - Missing values: Most frequent value imputation
       - Encoding: OneHotEncoder with handle_unknown='ignore'
         (guarantees inference won't crash on unseen categories in production)

    Parameters
    ----------
    numeric_features : Optional[List[str]]
        List of numeric feature column names.
    categorical_features : Optional[List[str]]
        List of categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor ready to fit on training data.
    """
    if numeric_features is None:
        numeric_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Pipeline for numerical features
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drop any unselected columns
    )

    return preprocessor


def split_features_and_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Housing DataFrame.
    target_col : str
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features) and y (target values).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop target column for X
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


if __name__ == "__main__":
    print("Testing data preprocessing pipeline...")
    data = load_dataset()
    print(f"Loaded raw dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

    data_cleaned, num_duplicates = remove_duplicates(data)
    print(f"Removed {num_duplicates} duplicate records. Clean shape: {data_cleaned.shape}")

    missing = check_missing_values(data_cleaned)
    print("\nMissing Values Summary:")
    print(missing[missing > 0])

    preprocessor = create_preprocessor()
    X, y = split_features_and_target(data_cleaned)
    X_processed = preprocessor.fit_transform(X)
    print(f"\nSuccessfully transformed feature matrix. Transformed shape: {X_processed.shape}")
    print("Preprocessing pipeline verified successfully!")
