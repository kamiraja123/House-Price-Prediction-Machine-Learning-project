"""
Model Training and Evaluation Module for House Price Prediction.

This module:
1. Loads and preprocesses the dataset.
2. Performs an 80/20 Train/Test split (preventing data leakage).
3. Trains and evaluates 4 regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Performs 5-Fold Cross-Validation on the training set.
5. Evaluates models on the unseen test set using MAE, MSE, RMSE, and R² Score.
6. Displays a comparative performance table.
7. Automatically selects the best model based on R² score and RMSE.
8. Saves the complete preprocessing + model pipeline to models/house_price_model.pkl.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import (
    load_dataset,
    remove_duplicates,
    split_features_and_target,
    create_preprocessor,
)


def get_models_dict() -> Dict[str, Any]:
    """
    Return dictionary of regression models with reproducible hyperparameters.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=12, min_samples_split=5, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=120, max_depth=16, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression evaluation metrics: MAE, MSE, RMSE, R² Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2,
    }


def train_and_compare_models(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Tuple[pd.DataFrame, str, Pipeline, Dict[str, Pipeline]]:
    """
    Train, cross-validate, and evaluate multiple regression pipelines.

    Returns
    -------
    Tuple[pd.DataFrame, str, Pipeline, Dict[str, Pipeline]]
        - Comparison DataFrame
        - Best model name
        - Best trained Pipeline
        - Dictionary of all trained pipelines
    """
    # 1. Load data & remove duplicates
    df = load_dataset(data_path)
    df_clean, dup_count = remove_duplicates(df)
    print(f"Loaded {len(df)} records. Dropped {dup_count} duplicates. Working with {len(df_clean)} records.")

    # 2. Split features and target
    X, y = split_features_and_target(df_clean)

    # 3. Train / Test Split (Strictly prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    models = get_models_dict()
    results = []
    trained_pipelines = {}

    print(f"\nTraining and evaluating {len(models)} models with {cv_folds}-fold cross validation...")
    print("-" * 75)

    for name, model in models.items():
        # Build unified Pipeline: preprocessor + model
        # Preprocessing fit is isolated to training data
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("regressor", model),
            ]
        )

        # 5-Fold Cross Validation on Training set
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
        )
        cv_r2_mean = float(cv_scores.mean())
        cv_r2_std = float(cv_scores.std())

        # Fit pipeline on training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        results.append({
            "Model": name,
            "CV R² (Mean)": cv_r2_mean,
            "CV R² (Std)": cv_r2_std,
            "Test MAE ($)": metrics["MAE"],
            "Test MSE": metrics["MSE"],
            "Test RMSE ($)": metrics["RMSE"],
            "Test R² Score": metrics["R2_Score"],
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Sort primarily by Test R² Score (descending), secondarily by RMSE (ascending)
    comparison_df = comparison_df.sort_values(
        by=["Test R² Score", "Test RMSE ($)"], ascending=[False, True]
    ).reset_index(drop=True)

    # Automatically select best performing model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 75)
    # Display formatted table
    formatted_df = comparison_df.copy()
    formatted_df["Test MAE ($)"] = formatted_df["Test MAE ($)"].map("${:,.2f}".format)
    formatted_df["Test RMSE ($)"] = formatted_df["Test RMSE ($)"].map("${:,.2f}".format)
    formatted_df["Test MSE"] = formatted_df["Test MSE"].map("{:,.2e}".format)
    formatted_df["Test R² Score"] = formatted_df["Test R² Score"].map("{:.4f}".format)
    formatted_df["CV R² (Mean)"] = formatted_df["CV R² (Mean)"].map("{:.4f}".format)
    formatted_df["CV R² (Std)"] = formatted_df["CV R² (Std)"].map("±{:.4f}".format)
    print(formatted_df.to_string(index=False))
    print("=" * 75)
    print(f"\n>>> Best Performing Model Selected: {best_model_name} <<<")
    best_r2 = comparison_df.iloc[0]["Test R² Score"]
    best_rmse = comparison_df.iloc[0]["Test RMSE ($)"]
    print(f"    R² Score: {best_r2:.4f} | RMSE: ${best_rmse:,.2f}")

    return comparison_df, best_model_name, best_pipeline, trained_pipelines


def save_best_model(
    pipeline: Pipeline,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    output_dir: str = None,
) -> str:
    """
    Save best trained pipeline, comparison metrics, and feature names to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted best Pipeline.
    best_model_name : str
        Name of best performing model.
    comparison_df : pd.DataFrame
        Model comparison dataframe.
    output_dir : str
        Directory to save models.

    Returns
    -------
    str
        Path to saved model file.
    """
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        output_dir = os.path.join(project_root, "models")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "house_price_model.pkl")

    # Save pipeline object using joblib
    joblib.dump(pipeline, model_path)
    print(f"\nSaved best model pipeline to: {model_path}")

    # Save comparison table CSV
    comparison_csv_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)

    # Save metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "best_r2_score": float(comparison_df.iloc[0]["Test R² Score"]),
        "best_rmse": float(comparison_df.iloc[0]["Test RMSE ($)"]),
        "best_mae": float(comparison_df.iloc[0]["Test MAE ($)"]),
        "models_evaluated": comparison_df["Model"].tolist(),
    }
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved comparison metrics to: {comparison_csv_path} and {metadata_path}")
    return model_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_file = os.path.join(project_root, "data", "house_data.csv")
    models_folder = os.path.join(project_root, "models")

    comparison, best_name, best_pipe, all_pipes = train_and_compare_models(data_file)
    save_best_model(best_pipe, best_name, comparison, models_folder)
