"""
Exploratory Data Analysis (EDA) Module for House Price Prediction.

Provides visualization functions for:
1. Price Distribution (Histogram + KDE)
2. Area vs Price (Scatter Plot + Trendline)
3. Bedrooms vs Price (Box Plot)
4. Bathrooms vs Price (Box Plot)
5. Correlation Heatmap (Numeric features correlation)
6. Location vs Average Price (Bar Chart)

Also provides a batch generation utility to save high-resolution figures.
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Setup consistent, modern aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_price_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of house prices with KDE and mean/median indicators.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Price"], kde=True, color="#3b82f6", bins=35, ax=ax, edgecolor="white", alpha=0.6)

    mean_price = df["Price"].mean()
    median_price = df["Price"].median()

    ax.axvline(mean_price, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: ${mean_price:,.0f}")
    ax.axvline(median_price, color="#10b981", linestyle="-.", linewidth=2, label=f"Median: ${median_price:,.0f}")

    ax.set_title("House Price Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("House Price ($)", fontsize=11)
    ax.set_ylabel("Count / Frequency", fontsize=11)
    ax.xaxis.set_major_formatter("${x:,.0f}")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_area_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot house area in square feet against house price with regression trendline.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Area",
        y="Price",
        hue="PropertyType",
        palette="viridis",
        alpha=0.65,
        s=45,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="Area",
        y="Price",
        scatter=False,
        ax=ax,
        color="#dc2626",
        line_kws={"linewidth": 2, "label": "Trendline"},
    )

    ax.set_title("Property Area vs. House Price", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Living Area (Square Feet)", fontsize=11)
    ax.set_ylabel("House Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(title="Property Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bedrooms_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot house price distribution grouped by number of bedrooms.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Bedrooms",
        y="Price",
        palette="Blues_r",
        hue="Bedrooms",
        legend=False,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    ax.set_title("Bedrooms vs. House Price", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Bedrooms", fontsize=11)
    ax.set_ylabel("House Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bathrooms_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot house price distribution grouped by number of bathrooms.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Bathrooms",
        y="Price",
        palette="Purples_r",
        hue="Bathrooms",
        legend=False,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    ax.set_title("Bathrooms vs. House Price", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Bathrooms", fontsize=11)
    ax.set_ylabel("House Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Compute and plot the Pearson correlation matrix for numerical features.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_location_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot average and median house price by geographic location.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    order = (
        df.groupby("Location")["Price"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.barplot(
        data=df,
        x="Location",
        y="Price",
        order=order,
        palette="mako",
        hue="Location",
        legend=False,
        errorbar=None,
        ax=ax,
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"${height:,.0f}",
                (p.get_x() + p.get_width() / 2.0, height / 2.0),
                ha="center",
                va="center",
                fontsize=9.5,
                color="white",
                fontweight="bold",
            )

    ax.set_title("Average House Price by Location", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Location", fontsize=11)
    ax.set_ylabel("Average Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    plt.xticks(rotation=15)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def generate_all_eda_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save all 6 required EDA plots to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating EDA plots in directory: {output_dir}")

    plot_price_distribution(df, os.path.join(output_dir, "01_price_distribution.png"))
    plot_area_vs_price(df, os.path.join(output_dir, "02_area_vs_price.png"))
    plot_bedrooms_vs_price(df, os.path.join(output_dir, "03_bedrooms_vs_price.png"))
    plot_bathrooms_vs_price(df, os.path.join(output_dir, "04_bathrooms_vs_price.png"))
    plot_correlation_heatmap(df, os.path.join(output_dir, "05_correlation_heatmap.png"))
    plot_location_vs_price(df, os.path.join(output_dir, "06_location_vs_price.png"))

    plt.close("all")
    print("All 6 EDA visualizations saved successfully!")


if __name__ == "__main__":
    from data_preprocessing import load_dataset, remove_duplicates

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    figures_dir = os.path.join(project_root, "reports", "figures")

    raw_data = load_dataset()
    clean_df, _ = remove_duplicates(raw_data)
    generate_all_eda_plots(clean_df, figures_dir)
