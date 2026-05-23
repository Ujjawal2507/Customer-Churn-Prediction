"""
Churn EDA — Exploratory Data Analysis
=======================================
Standalone EDA module. Run independently or import into notebooks.
Produces charts saved to outputs/eda/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


PALETTE = {"No Churn": "#378ADD", "Churn": "#D85A30"}
OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                         "axes.spines.right": False})


def plot_churn_distribution(df: pd.DataFrame):
    counts = df["Churn"].value_counts()
    labels = ["No Churn", "Churn"]
    colors = [PALETTE["No Churn"], PALETTE["Churn"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar
    axes[0].bar(labels, counts.values, color=colors, width=0.5)
    axes[0].set_title("Churn Count")
    axes[0].set_ylabel("Customers")

    # Pie
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Churn Share")

    fig.suptitle("Churn Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/churn_distribution.png")
    plt.close()
    print(f"[✓] {OUTPUT_DIR}/churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    available   = [c for c in numeric_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        churn_vals    = df[df["Churn"] == "Yes"][col].dropna()
        no_churn_vals = df[df["Churn"] == "No"][col].dropna()
        ax.hist(no_churn_vals, bins=30, alpha=0.6, color=PALETTE["No Churn"], label="No Churn")
        ax.hist(churn_vals,    bins=30, alpha=0.6, color=PALETTE["Churn"],    label="Churn")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend()

    fig.suptitle("Numeric Feature Distributions by Churn", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/numeric_distributions.png")
    plt.close()
    print(f"[✓] {OUTPUT_DIR}/numeric_distributions.png")


def plot_categorical_churn(df: pd.DataFrame, cols: list = None):
    if cols is None:
        cols = ["Contract", "InternetService", "PaymentMethod", "tenure_group"]

    if "tenure_group" not in df.columns:
        df = df.copy()
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"]
        )

    available = [c for c in cols if c in df.columns]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        ct = df.groupby(col)["Churn"].value_counts(normalize=True).unstack().fillna(0)
        if "Yes" in ct.columns:
            ct["Yes"].sort_values().plot(kind="barh", ax=ax, color=PALETTE["Churn"])
            ax.set_title(f"Churn Rate by {col}")
            ax.set_xlabel("Churn Rate")
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.suptitle("Churn Rate by Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/categorical_churn_rates.png")
    plt.close()
    print(f"[✓] {OUTPUT_DIR}/categorical_churn_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()
    print(f"[✓] {OUTPUT_DIR}/correlation_heatmap.png")


def run_full_eda(filepath: str):
    set_style()
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded {df.shape[0]:,} rows")

    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_churn(df)
    plot_correlation_heatmap(df)
    print(f"\n[✓] All EDA charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    run_full_eda(path)
