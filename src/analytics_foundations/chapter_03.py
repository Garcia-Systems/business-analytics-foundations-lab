"""Transparent summation and aggregation calculations for Chapter 3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_03_data


def add_transaction_revenue(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with revenue calculated as quantity times unit price."""

    result = data.copy()
    result["revenue"] = result["quantity"] * result["unit_price"]
    return result


def arithmetic_mean(values: ArrayLike) -> float:
    """Return the arithmetic mean, explicitly as sum divided by count."""

    observations = np.asarray(values, dtype=float)
    if observations.size == 0:
        raise ValueError("at least one observation is required")
    return float(np.sum(observations) / observations.size)


def weighted_average(prices: ArrayLike, quantities: ArrayLike) -> float:
    """Return average revenue per unit using quantities as weights."""

    price_values = np.asarray(prices, dtype=float)
    quantity_values = np.asarray(quantities, dtype=float)
    if price_values.shape != quantity_values.shape:
        raise ValueError("prices and quantities must have the same shape")
    total_quantity = np.sum(quantity_values)
    if total_quantity == 0:
        raise ValueError("total quantity must not be zero")
    return float(np.sum(price_values * quantity_values) / total_quantity)


def calculate_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the chapter's overall transaction metrics."""

    total_revenue = float(data["revenue"].sum())
    total_quantity = int(data["quantity"].sum())
    return {
        "total_revenue": total_revenue,
        "average_transaction_revenue": arithmetic_mean(data["revenue"]),
        "total_quantity": total_quantity,
        "simple_average_unit_price": arithmetic_mean(data["unit_price"]),
        "weighted_average_unit_price": weighted_average(
            data["unit_price"], data["quantity"]
        ),
    }


def category_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and quantity, then calculate revenue contribution."""

    summary = data.groupby("category", sort=False).agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum")
    )
    summary["revenue_share_pct"] = summary["revenue"] / data["revenue"].sum() * 100
    return summary


def create_revenue_by_category_figure(summary: pd.DataFrame, output_path: Path) -> Path:
    """Save a bar chart whose heights are grouped revenue sums."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    summary["revenue"].plot.bar(ax=axis, color="tab:blue")
    axis.set(
        title="Riverside Cafe revenue by category",
        xlabel="Category",
        ylabel="Revenue ($)",
    )
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def create_contribution_figure(summary: pd.DataFrame, output_path: Path) -> Path:
    """Save a bar chart of each category's percentage contribution."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    summary["revenue_share_pct"].plot.bar(ax=axis, color="tab:orange")
    axis.set(
        title="Contribution to total revenue",
        xlabel="Category",
        ylabel="Share of revenue (%)",
        ylim=(0, max(50, summary["revenue_share_pct"].max() * 1.15)),
    )
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run(output_dir: Path | None = None) -> int:
    """Run and narrate the Chapter 3 transaction experiment."""

    data = add_transaction_revenue(load_chapter_03_data())
    revenues = data["revenue"].to_numpy()
    loop_total = 0.0
    for value in revenues:
        loop_total += value
    numpy_total = float(np.sum(revenues))
    pandas_total = float(data["revenue"].sum())
    metrics = calculate_metrics(data)
    summary = category_summary(data)

    print("Chapter 3 — Summation & Aggregation")
    print("Question: What did Riverside Cafe sell, earn, and earn per unit?\n")
    print("First 5 transactions:")
    print(
        data.head().to_string(
            index=False,
            formatters={
                "unit_price": "${:,.2f}".format,
                "revenue": "${:,.2f}".format,
            },
        )
    )
    print("\nOne summation, three implementations:")
    print(f"  loop total = NumPy total = pandas total = ${loop_total:,.2f}")
    assert loop_total == numpy_total == pandas_total
    print("\nBusiness metrics:")
    print(f"  Total revenue:                     ${metrics['total_revenue']:,.2f}")
    print(f"  Average revenue per transaction:   ${metrics['average_transaction_revenue']:,.2f}")
    print(f"  Total quantity:                    {metrics['total_quantity']}")
    print(f"  Simple average listed price:       ${metrics['simple_average_unit_price']:,.2f}")
    print(f"  Quantity-weighted price per unit:  ${metrics['weighted_average_unit_price']:,.2f}")
    print("\nGrouped summation (rows → category sums → bar heights):")
    print(summary.round(2).to_string())

    destination = output_dir or PROJECT_ROOT / "figures"
    revenue_path = create_revenue_by_category_figure(
        summary, destination / "chapter-03-revenue-by-category.png"
    )
    share_path = create_contribution_figure(
        summary, destination / "chapter-03-category-contributions.png"
    )
    revenue_leader = summary["revenue"].idxmax()
    quantity_leader = summary["quantity"].idxmax()
    price_leader = data.groupby("category")["unit_price"].max().idxmax()
    print(f"\nInterpretation: {revenue_leader} generated the largest total revenue; ")
    print(
        f"{quantity_leader} sold the most units; "
        f"{price_leader} had the highest single price."
    )
    print(
        "These metrics answer different questions: revenue reflects quantity, price, or both."
    )
    print(f"Figures saved to: {revenue_path} and {share_path}")
    return 0
