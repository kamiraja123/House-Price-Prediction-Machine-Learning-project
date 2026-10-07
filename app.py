"""
House Price Prediction Web Application.
Built with Streamlit and Scikit-learn.

Provides:
- Interactive inputs for all property features (Area, Bedrooms, Bathrooms, Floors, Parking, Location, Year Built, Property Type).
- Model inference with instant price prediction and confidence range.
- Model performance comparison view (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
- Exploratory Data Analysis (EDA) charts and market insights.
- Preset property configurations for rapid testing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add src folder to sys.path dynamically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import HousePricePredictor, REQUIRED_FEATURES
from data_preprocessing import load_dataset, remove_duplicates
from eda import (
    plot_price_distribution,
    plot_area_vs_price,
    plot_bedrooms_vs_price,
    plot_bathrooms_vs_price,
    plot_correlation_heatmap,
    plot_location_vs_price,
)
from train_model import train_and_compare_models, save_best_model


# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header typography */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Price Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
        margin: 1.2rem 0;
        text-align: center;
    }
    .prediction-title {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .prediction-subtext {
        font-size: 0.95rem;
        opacity: 0.88;
    }

    /* Feature summary tag badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 1rem;
    }
    .badge-pill {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Dynamic Paths
# ---------------------------------------------------------
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "house_price_model.pkl")
METADATA_PATH = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
COMPARISON_PATH = os.path.join(CURRENT_DIR, "models", "model_comparison.csv")
DATA_PATH = os.path.join(CURRENT_DIR, "data", "house_data.csv")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor():
    """Load and cache the HousePricePredictor."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return HousePricePredictor(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_cleaned_data():
    """Load and cache the dataset with duplicates removed."""
    if not os.path.exists(DATA_PATH):
        return None
    raw = load_dataset(DATA_PATH)
    clean, _ = remove_duplicates(raw)
    return clean


@st.cache_data(show_spinner=False)
def get_model_metadata():
    """Load metadata about the best trained model."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def get_comparison_table():
    """Load the model comparison metrics table."""
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)
    return None


def render_dataframe(df: pd.DataFrame, hide_index: bool = True):
    """Render dataframe cleanly supporting both legacy and modern Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except Exception:
        try:
            st.dataframe(df, use_container_width=True, hide_index=hide_index)
        except Exception:
            st.dataframe(df)


# ---------------------------------------------------------
# Sidebar: Status & Presets
# ---------------------------------------------------------
with st.sidebar:
    try:
        st.image(
            "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=600&auto=format&fit=crop&q=80",
            width="stretch",
        )
    except Exception:
        st.image(
            "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=600&auto=format&fit=crop&q=80"
        )
    st.title("🏡 Valuation AI")
    st.caption("Machine Learning House Price Estimator")

    st.markdown("---")
    st.subheader("🤖 Model Status")

    metadata = get_model_metadata()
    if os.path.exists(MODEL_PATH) and metadata:
        st.success(f"**Model:** {metadata.get('best_model_name', 'Trained')}")
        st.metric("Test R² Score", f"{metadata.get('best_r2_score', 0.0):.4f}")
        st.metric("Test RMSE", f"${metadata.get('best_rmse', 0.0):,.0f}")
    else:
        st.warning("Model not found on disk.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training models and saving best pipeline..."):
                comp, best_name, best_p, _ = train_and_compare_models(DATA_PATH)
                save_best_model(best_p, best_name, comp, os.path.join(CURRENT_DIR, "models"))
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    st.markdown("---")
    st.subheader("⚡ Quick Presets")
    st.caption("Click a preset to populate property details:")

    preset_choice = st.radio(
        "Select Sample Property:",
        [
            "Custom Input",
            "Modern Downtown Condo",
            "Suburban Family Home",
            "Luxury Waterfront Villa",
            "Cozy Starter Townhouse",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        **Features included:**
        - Area (Square Feet)
        - Bedrooms & Bathrooms
        - Floors & Parking
        - Location & Property Type
        - Year Built
        """
    )


# Set preset defaults
presets = {
    "Custom Input": {
        "area": 2000.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 2,
        "parking": 2,
        "location": "Suburban",
        "year_built": 2015,
        "property_type": "Single Family",
    },
    "Modern Downtown Condo": {
        "area": 1150.0,
        "bedrooms": 2,
        "bathrooms": 2,
        "floors": 1,
        "parking": 1,
        "location": "Downtown",
        "year_built": 2021,
        "property_type": "Condo",
    },
    "Suburban Family Home": {
        "area": 2650.0,
        "bedrooms": 4,
        "bathrooms": 3,
        "floors": 2,
        "parking": 2,
        "location": "Suburban",
        "year_built": 2012,
        "property_type": "Single Family",
    },
    "Luxury Waterfront Villa": {
        "area": 4800.0,
        "bedrooms": 5,
        "bathrooms": 5,
        "floors": 3,
        "parking": 4,
        "location": "Waterfront",
        "year_built": 2022,
        "property_type": "Villa",
    },
    "Cozy Starter Townhouse": {
        "area": 1400.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 2,
        "parking": 1,
        "location": "Urban",
        "year_built": 2008,
        "property_type": "Townhouse",
    },
}

active_preset = presets[preset_choice]


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="hero-title">🏡 House Price Prediction AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Intelligent property valuation powered by Scikit-Learn regression pipelines. Enter property specifications to estimate real market value.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 Model Comparison", "📈 Market Insights & EDA"])


# ---------------------------------------------------------
# Tab 1: Price Prediction
# ---------------------------------------------------------
with tab1:
    st.subheader("Property Characteristics")
    st.markdown("Configure the property features below:")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📐 Structural Specifications")
        area_input = st.number_input(
            "Living Area (Square Feet)",
            min_value=300.0,
            max_value=10000.0,
            value=float(active_preset["area"]),
            step=50.0,
            help="Total heated living area in square feet.",
        )

        bedrooms_input = st.selectbox(
            "Number of Bedrooms",
            options=[1, 2, 3, 4, 5, 6],
            index=[1, 2, 3, 4, 5, 6].index(active_preset["bedrooms"]),
            help="Total number of bedrooms in the property.",
        )

        bathrooms_input = st.selectbox(
            "Number of Bathrooms",
            options=[1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(active_preset["bathrooms"]),
            help="Total number of full and three-quarter bathrooms.",
        )

        floors_input = st.selectbox(
            "Floors / Stories",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(active_preset["floors"]),
            help="Number of above-ground floors.",
        )

    with col2:
        st.markdown("#### 📍 Location & Property Details")

        parking_input = st.selectbox(
            "Parking / Garage Spaces",
            options=[0, 1, 2, 3, 4],
            index=[0, 1, 2, 3, 4].index(active_preset["parking"]),
            help="Number of dedicated covered or garage parking spots.",
        )

        location_options = ["Downtown", "Suburban", "Urban", "Rural", "Waterfront", "Uptown"]
        location_input = st.selectbox(
            "Geographic Location",
            options=location_options,
            index=location_options.index(active_preset["location"]),
            help="Property neighborhood or zone.",
        )

        year_built_input = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2026,
            value=int(active_preset["year_built"]),
            step=1,
            help="Year the construction of the house was completed.",
        )

        property_type_options = ["Single Family", "Townhouse", "Apartment", "Condo", "Villa"]
        property_type_input = st.selectbox(
            "Property Type",
            options=property_type_options,
            index=property_type_options.index(active_preset["property_type"]),
            help="Architectural classification of the residence.",
        )

    st.markdown("---")

    # Predict Button
    predict_btn = st.button("🔮 Predict House Price", type="primary", use_container_width=True)

    if predict_btn:
        predictor = get_predictor()

        if predictor is None:
            st.error("Model file not found! Please train the model using the button in the sidebar.")
        else:
            with st.spinner("Calculating estimated valuation..."):
                input_payload = {
                    "Area": float(area_input),
                    "Bedrooms": int(bedrooms_input),
                    "Bathrooms": int(bathrooms_input),
                    "Floors": int(floors_input),
                    "Parking": int(parking_input),
                    "Location": str(location_input),
                    "YearBuilt": int(year_built_input),
                    "PropertyType": str(property_type_input),
                }

                try:
                    result = predictor.predict(input_payload)

                    # Highlighted prediction card
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div class="prediction-title">Valuation Result</div>
                            <div class="prediction-value">Estimated House Price: {result['formatted_price']}</div>
                            <div class="prediction-subtext">
                                Estimated Market Range: <b>{result['formatted_range']}</b> (±6% Variance)
                            </div>
                            <div class="badge-container">
                                <span class="badge-pill">📐 {area_input:,.0f} sq. ft.</span>
                                <span class="badge-pill">🛏️ {bedrooms_input} Beds</span>
                                <span class="badge-pill">🛁 {bathrooms_input} Baths</span>
                                <span class="badge-pill">🚗 {parking_input} Parking</span>
                                <span class="badge-pill">📍 {location_input}</span>
                                <span class="badge-pill">🏛️ {property_type_input}</span>
                                <span class="badge-pill">📅 Built {year_built_input}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Key Valuation Metric Cards
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric(
                            label="Price per Sq. Ft.",
                            value=f"${result['price_per_sqft']:,.2f}",
                            help="Estimated valuation divided by total living area.",
                        )
                    with m2:
                        st.metric(
                            label="Conservative Estimate",
                            value=f"${result['lower_bound']:,.0f}",
                            help="Lower bound valuation.",
                        )
                    with m3:
                        st.metric(
                            label="Optimistic Estimate",
                            value=f"${result['upper_bound']:,.0f}",
                            help="Upper bound valuation.",
                        )
                    with m4:
                        age = 2026 - int(year_built_input)
                        st.metric(
                            label="Property Age",
                            value=f"{age} Years",
                            help="Calculated relative to 2026.",
                        )

                except Exception as ex:
                    st.error(f"Prediction failed with error: {str(ex)}")


# ---------------------------------------------------------
# Tab 2: Model Comparison & Evaluation
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Performance & Comparison")
    st.markdown(
        "Four regression models were trained using 5-Fold Cross Validation and evaluated on an unseen 20% test split."
    )

    comparison_df = get_comparison_table()
    if comparison_df is not None:
        # Format metrics table nicely for display
        display_df = comparison_df.copy()
        display_df["Test MAE"] = display_df["Test MAE ($)"].map("${:,.2f}".format)
        display_df["Test RMSE"] = display_df["Test RMSE ($)"].map("${:,.2f}".format)
        display_df["Test MSE"] = display_df["Test MSE"].map("{:,.2e}".format)
        display_df["Test R²"] = display_df["Test R² Score"].map("{:.4f}".format)
        display_df["CV R² (Mean)"] = display_df["CV R² (Mean)"].map("{:.4f}".format)
        display_df["CV R² (Std)"] = display_df["CV R² (Std)"].map("±{:.4f}".format)

        cols_to_show = [
            "Model",
            "Test R²",
            "Test RMSE",
            "Test MAE",
            "CV R² (Mean)",
            "CV R² (Std)",
            "Test MSE",
        ]
        render_dataframe(display_df[cols_to_show], hide_index=True)

        best_row = comparison_df.iloc[0]
        st.success(
            f"🏆 **Winner: {best_row['Model']}** achieved the highest R² score of "
            f"**{best_row['Test R² Score']:.4f}** and the lowest RMSE of **${best_row['Test RMSE ($)']:,.2f}**."
        )

        st.markdown("#### 📊 Comparative Metric Visualizations")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_r2, ax_r2 = plt.subplots(figsize=(6, 3.8))
            colors = ["#10b981" if m == best_row["Model"] else "#94a3b8" for m in comparison_df["Model"]]
            bars = ax_r2.barh(comparison_df["Model"], comparison_df["Test R² Score"], color=colors)
            ax_r2.set_xlim(0.85, 1.0)
            ax_r2.set_xlabel("R² Score (Higher is better)")
            ax_r2.set_title("Test R² Score by Model", fontweight="bold")
            for bar in bars:
                w = bar.get_width()
                ax_r2.text(w - 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", color="white", fontweight="bold")
            fig_r2.tight_layout()
            st.pyplot(fig_r2)
            plt.close(fig_r2)

        with col_c2:
            fig_rmse, ax_rmse = plt.subplots(figsize=(6, 3.8))
            colors_rmse = ["#3b82f6" if m == best_row["Model"] else "#cbd5e1" for m in comparison_df["Model"]]
            bars_rmse = ax_rmse.barh(comparison_df["Model"], comparison_df["Test RMSE ($)"], color=colors_rmse)
            ax_rmse.set_xlabel("Root Mean Squared Error ($) (Lower is better)")
            ax_rmse.set_title("Test RMSE by Model", fontweight="bold")
            ax_rmse.xaxis.set_major_formatter("${x:,.0f}")
            for bar in bars_rmse:
                w = bar.get_width()
                ax_rmse.text(w * 0.6, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", color="white", fontweight="bold")
            fig_rmse.tight_layout()
            st.pyplot(fig_rmse)
            plt.close(fig_rmse)

    else:
        st.info("Comparison table not found. Please train models to view performance comparisons.")


# ---------------------------------------------------------
# Tab 3: Market Insights & EDA
# ---------------------------------------------------------
with tab3:
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("Gain intuition into property pricing factors, correlation dynamics, and regional distributions.")

    df_clean = get_cleaned_data()
    if df_clean is not None:
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📊 Pricing & Area", "🛏️ Rooms & Bathrooms", "🗺️ Location & Heatmap"])

        with eda_tab1:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("##### 1. Price Distribution")
                fig1 = plot_price_distribution(df_clean)
                st.pyplot(fig1)
                plt.close(fig1)
            with col_e2:
                st.markdown("##### 2. Living Area vs. Price")
                fig2 = plot_area_vs_price(df_clean)
                st.pyplot(fig2)
                plt.close(fig2)

        with eda_tab2:
            col_e3, col_e4 = st.columns(2)
            with col_e3:
                st.markdown("##### 3. Bedrooms vs. Price")
                fig3 = plot_bedrooms_vs_price(df_clean)
                st.pyplot(fig3)
                plt.close(fig3)
            with col_e4:
                st.markdown("##### 4. Bathrooms vs. Price")
                fig4 = plot_bathrooms_vs_price(df_clean)
                st.pyplot(fig4)
                plt.close(fig4)

        with eda_tab3:
            col_e5, col_e6 = st.columns(2)
            with col_e5:
                st.markdown("##### 5. Correlation Heatmap")
                fig5 = plot_correlation_heatmap(df_clean)
                st.pyplot(fig5)
                plt.close(fig5)
            with col_e6:
                st.markdown("##### 6. Location vs. Average Price")
                fig6 = plot_location_vs_price(df_clean)
                st.pyplot(fig6)
                plt.close(fig6)

        with st.expander("🔍 Explore Raw Dataset (First 50 Rows)"):
            render_dataframe(df_clean.head(50), hide_index=False)
    else:
        st.warning("Dataset not found at data/house_data.csv")
