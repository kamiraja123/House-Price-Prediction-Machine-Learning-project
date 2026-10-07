"""
Dataset generator script for House Price Prediction.
Generates a realistic, multifaceted housing dataset with continuous, discrete,
and categorical features, incorporating realistic economic pricing relationships,
realistic market noise, slight duplicates, and missing values to thoroughly exercise
preprocessing and modeling pipelines.
"""

import os
import numpy as np
import pandas as pd


def generate_housing_data(
    n_samples: int = 3000,
    random_state: int = 42,
    include_imperfections: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic house price dataset with realistic distributions and relationships.

    Parameters
    ----------
    n_samples : int
        Number of house records to generate.
    random_state : int
        Seed for reproducibility.
    include_imperfections : bool
        Whether to introduce realistic duplicates and missing values.

    Returns
    -------
    pd.DataFrame
        Generated housing dataset.
    """
    rng = np.random.default_rng(random_state)

    # 1. Property Types & Location categories
    property_types = ["Single Family", "Townhouse", "Apartment", "Condo", "Villa"]
    property_weights = [0.45, 0.20, 0.15, 0.12, 0.08]

    locations = ["Downtown", "Suburban", "Urban", "Rural", "Waterfront", "Uptown"]
    location_weights = [0.22, 0.35, 0.18, 0.10, 0.07, 0.08]

    chosen_property_types = rng.choice(property_types, size=n_samples, p=property_weights)
    chosen_locations = rng.choice(locations, size=n_samples, p=location_weights)

    # 2. Area in Square Feet (dependent partly on property type)
    base_areas = {
        "Apartment": (500, 1400),
        "Condo": (650, 1800),
        "Townhouse": (1100, 2600),
        "Single Family": (1400, 4200),
        "Villa": (2800, 5800),
    }

    area = np.zeros(n_samples, dtype=float)
    for i, p_type in enumerate(chosen_property_types):
        min_a, max_a = base_areas[p_type]
        area[i] = rng.uniform(min_a, max_a)
    area = np.round(area, -1)  # Round to nearest 10 sq ft

    # 3. Bedrooms (correlated with area)
    # Area ~ 500 sq ft per bedroom roughly
    bedrooms = np.clip(np.round(area / 650 + rng.normal(0, 0.5, n_samples)), 1, 6).astype(int)

    # 4. Bathrooms (correlated with bedrooms)
    bathrooms = np.clip(np.round(bedrooms * 0.75 + rng.normal(0, 0.4, n_samples)), 1, 5).astype(int)

    # 5. Floors
    floors = np.zeros(n_samples, dtype=int)
    for i, p_type in enumerate(chosen_property_types):
        if p_type in ["Apartment", "Condo"]:
            floors[i] = rng.choice([1, 2], p=[0.85, 0.15])
        elif p_type == "Townhouse":
            floors[i] = rng.choice([2, 3], p=[0.70, 0.30])
        elif p_type == "Villa":
            floors[i] = rng.choice([1, 2, 3], p=[0.20, 0.60, 0.20])
        else:
            floors[i] = rng.choice([1, 2, 3, 4], p=[0.35, 0.50, 0.12, 0.03])

    # 6. Parking Spaces (0 to 4)
    parking = np.clip(np.round(bedrooms * 0.4 + rng.normal(0.5, 0.7, n_samples)), 0, 4).astype(int)

    # 7. Year Built (1960 to 2024)
    year_built = rng.integers(1960, 2025, size=n_samples)

    # 8. Realistic Economic Pricing Formula
    # Base valuation
    base_price = 85000.0

    # Square footage valuation ($135 to $220/sq ft depending on location)
    location_multiplier = {
        "Waterfront": 1.78,
        "Downtown": 1.48,
        "Uptown": 1.32,
        "Urban": 1.15,
        "Suburban": 1.00,
        "Rural": 0.78,
    }

    property_multiplier = {
        "Villa": 1.35,
        "Single Family": 1.12,
        "Townhouse": 0.98,
        "Condo": 0.92,
        "Apartment": 0.84,
    }

    # Linear feature additions
    price = (
        base_price
        + (area * 145.0)
        + (bedrooms * 22000.0)
        + (bathrooms * 19000.0)
        + (floors * 12500.0)
        + (parking * 15000.0)
        + ((year_built - 1960) * 1200.0)
    )

    # Apply location and property type multipliers
    for i in range(n_samples):
        loc = chosen_locations[i]
        ptype = chosen_property_types[i]
        price[i] *= location_multiplier[loc] * property_multiplier[ptype]

    # Add realistic market noise (~5%)
    noise = rng.normal(1.0, 0.05, n_samples)
    price = np.round(price * noise, -2)  # Round to nearest 100

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Parking": parking,
        "Location": chosen_locations,
        "YearBuilt": year_built,
        "PropertyType": chosen_property_types,
        "Price": price,
    })

    if include_imperfections:
        # Introduce a few realistic missing values (~1-2% for Parking and YearBuilt)
        missing_parking_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
        missing_year_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
        df.loc[missing_parking_idx, "Parking"] = np.nan
        df.loc[missing_year_idx, "YearBuilt"] = np.nan

        # Introduce a few realistic duplicate rows (~15 duplicates) to test duplicate removal
        duplicate_indices = rng.choice(n_samples, size=15, replace=False)
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "house_data.csv")
    print(f"Generating realistic house price dataset to: {csv_path}...")
    dataset = generate_housing_data(n_samples=3000, random_state=42)
    dataset.to_csv(csv_path, index=False)
    print(f"Successfully generated {len(dataset)} rows and {len(dataset.columns)} columns.")
    print(dataset.head())
