"""
Prediction Inference Module for House Price Prediction.

Provides:
- HousePricePredictor class to load saved model pipeline and run predictions.
- Robust input validation and edge-case handling.
- Valuation confidence intervals and price-per-square-foot calculations.
- CLI demonstration interface.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional


REQUIRED_FEATURES = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Parking",
    "Location",
    "YearBuilt",
    "PropertyType",
]

DEFAULT_VALUES = {
    "Area": 1800.0,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Floors": 1,
    "Parking": 1,
    "Location": "Suburban",
    "YearBuilt": 2005,
    "PropertyType": "Single Family",
}


def get_default_model_path() -> str:
    """
    Get dynamic path to saved model pipeline.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    return os.path.join(project_root, "models", "house_price_model.pkl")


class HousePricePredictor:
    """
    Inference class to handle model loading, data validation, and price prediction.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor with the trained pipeline file.
        """
        if model_path is None:
            model_path = get_default_model_path()

        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the joblib pipeline from disk safely.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}. "
                "Please train the model first by running src/train_model.py"
            )
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def validate_and_prepare_input(
        self, input_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Validate, sanitize, and convert input data into an appropriate pandas DataFrame.

        Parameters
        ----------
        input_data : Union[Dict[str, Any], pd.DataFrame]
            Single property features or multiple property DataFrame.

        Returns
        -------
        pd.DataFrame
            Sanitized DataFrame matching training feature schema.
        """
        if isinstance(input_data, dict):
            # Check and fill missing keys with safe defaults
            sanitized = {}
            for col in REQUIRED_FEATURES:
                val = input_data.get(col, DEFAULT_VALUES[col])
                # Validate numeric fields
                if col in ["Area", "Bedrooms", "Bathrooms", "Floors", "Parking", "YearBuilt"]:
                    try:
                        val = float(val) if col == "Area" else int(val)
                    except (ValueError, TypeError):
                        val = DEFAULT_VALUES[col]
                # Validate string fields
                elif col in ["Location", "PropertyType"]:
                    if not isinstance(val, str) or not val.strip():
                        val = DEFAULT_VALUES[col]
                    val = str(val).strip()

                sanitized[col] = val

            df = pd.DataFrame([sanitized])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            for col in REQUIRED_FEATURES:
                if col not in df.columns:
                    df[col] = DEFAULT_VALUES[col]
            df = df[REQUIRED_FEATURES]
        else:
            raise TypeError("input_data must be either a Python dict or a pandas DataFrame.")

        return df

    def predict(
        self, input_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Generate price prediction and valuation insights.

        Parameters
        ----------
        input_data : Union[Dict[str, Any], pd.DataFrame]
            Input property features.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing predicted price, formatted string,
            estimated price range (confidence interval), and price per sqft.
        """
        df_prepared = self.validate_and_prepare_input(input_data)
        raw_predictions = self.model.predict(df_prepared)

        # For single record requests (the typical app workflow)
        if len(raw_predictions) == 1:
            predicted_price = float(raw_predictions[0])
            # Ensure price cannot be negative
            predicted_price = max(10000.0, predicted_price)

            area = float(df_prepared.iloc[0]["Area"])
            price_per_sqft = (predicted_price / area) if area > 0 else 0.0

            # Estimate ±6% standard market valuation variance range
            lower_bound = predicted_price * 0.94
            upper_bound = predicted_price * 1.06

            return {
                "predicted_price": round(predicted_price, 2),
                "formatted_price": f"${predicted_price:,.2f}",
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "formatted_range": f"${lower_bound:,.0f} - ${upper_bound:,.0f}",
                "price_per_sqft": round(price_per_sqft, 2),
                "input_summary": df_prepared.iloc[0].to_dict(),
            }

        # Multi-record batch predictions
        return {
            "predicted_prices": [max(10000.0, float(p)) for p in raw_predictions],
            "formatted_prices": [f"${max(10000.0, float(p)):,.2f}" for p in raw_predictions],
            "count": len(raw_predictions),
        }


if __name__ == "__main__":
    print("Testing HousePricePredictor on sample property...")
    predictor = HousePricePredictor()

    sample_house = {
        "Area": 2500,
        "Bedrooms": 4,
        "Bathrooms": 3,
        "Floors": 2,
        "Parking": 2,
        "Location": "Downtown",
        "YearBuilt": 2018,
        "PropertyType": "Single Family",
    }

    result = predictor.predict(sample_house)
    print("\n--- Valuation Report ---")
    print(f"Property Features: {result['input_summary']}")
    print(f"Estimated House Price: {result['formatted_price']}")
    print(f"Estimated Price Range: {result['formatted_range']}")
    print(f"Price per Sq. Ft.: ${result['price_per_sqft']:,.2f}")
