"""Executable additive and multiplicative growth models for Chapter 2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from analytics_foundations.datasets import PROJECT_ROOT

STARTING_CUSTOMERS = 500
MONTHLY_ADDITION = 50
MONTHLY_GROWTH_RATE = 0.08


def linear_customers(months: ArrayLike) -> NDArray[np.number] | float:
    """Return customers when the business adds 50 customers per month."""

    return STARTING_CUSTOMERS + MONTHLY_ADDITION * np.asarray(months)


def exponential_customers(months: ArrayLike) -> NDArray[np.number] | float:
    """Return customers when the business compounds at 8% per month."""

    return STARTING_CUSTOMERS * np.power(1 + MONTHLY_GROWTH_RATE, months)


def percentage_change(old: ArrayLike, new: ArrayLike) -> NDArray[np.number] | float:
    """Return relative change from ``old`` to ``new`` as a percentage."""

    old_values = np.asarray(old)
    if np.any(old_values == 0):
        raise ValueError("percentage change is undefined when the old value is zero")
    return (np.asarray(new) - old_values) / old_values * 100


def percentage_point_change(old_rate: ArrayLike, new_rate: ArrayLike):
    """Return percentage-point change for rates expressed as decimals."""

    return (np.asarray(new_rate) - np.asarray(old_rate)) * 100


def doubling_time(growth_rate: float) -> float:
    """Return periods needed to double under a positive compound growth rate."""

    if growth_rate <= 0:
        raise ValueError("growth rate must be positive to calculate doubling time")
    return float(np.log(2) / np.log(1 + growth_rate))


def crossover_month(max_months: int = 1_000) -> int:
    """Return the first positive whole month exponential exceeds linear growth."""

    if max_months < 1:
        raise ValueError("max_months must be at least 1")
    months = np.arange(1, max_months + 1)
    crossings = months[exponential_customers(months) > linear_customers(months)]
    if crossings.size == 0:
        raise ValueError(f"no crossover found within {max_months} months")
    return int(crossings[0])


def create_growth_figure(months: ArrayLike, output_path: Path) -> Path:
    """Plot the chapter's linear and exponential customer models."""

    months = np.asarray(months)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(months, linear_customers(months), label="Linear: add 50/month")
    axis.plot(months, exponential_customers(months), label="Exponential: grow 8%/month")
    crossing = crossover_month()
    axis.scatter([crossing], [exponential_customers(crossing)], color="black", zorder=3)
    axis.annotate(
        f"Exponential overtakes\nat month {crossing}",
        xy=(crossing, exponential_customers(crossing)),
        xytext=(25, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axis.set(title="James River Analytics customer growth", xlabel="Months", ylabel="Customers")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def create_repeated_multiplication_figure(months: ArrayLike, output_path: Path) -> Path:
    """Show that a fixed rate creates increasingly large absolute additions."""

    months = np.asarray(months)
    customers = exponential_customers(months)
    additions = np.diff(customers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(months, customers, marker="o", color="tab:blue")
    top.set(ylabel="Customers", title="Repeated multiplication by 1.08")
    top.grid(alpha=0.25)
    bottom.bar(months[1:], additions, color="tab:orange")
    bottom.set(
        xlabel="Month",
        ylabel="Customers added",
        title="The percentage stays 8%, but the absolute addition grows",
    )
    bottom.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run(output_dir: Path | None = None) -> int:
    """Run and narrate the Chapter 2 growth experiment."""

    print("Chapter 2 — Exponents, Logs & Growth")
    print("Scenario: James River Analytics begins with 500 subscribers.\n")
    print("Models:")
    print("  Linear:      N(t) = 500 + 50t")
    print("  Exponential: N(t) = 500(1.08)^t")
    print("\n Month   Linear   Exponential")
    for month in (0, 1, 2, 3, 6, 12, 24):
        print(f" {month:5d}  {linear_customers(month):7.0f}  {exponential_customers(month):12.1f}")

    growth = percentage_change(exponential_customers(0), exponential_customers(1))
    double = doubling_time(MONTHLY_GROWTH_RATE)
    crossing = crossover_month()
    print(f"\nMonthly exponential growth: {growth:.1f}%")
    print(f"Doubling time: ln(2) / ln(1.08) = {double:.2f} months")
    print(f"First whole month exponential exceeds linear: {crossing}")

    destination = output_dir or PROJECT_ROOT / "figures"
    growth_path = create_growth_figure(
        np.arange(0, 25), destination / "chapter-02-linear-vs-exponential.png"
    )
    multiplication_path = create_repeated_multiplication_figure(
        np.arange(0, 13), destination / "chapter-02-repeated-multiplication.png"
    )
    print("\nInterpretation: Linear change repeatedly adds; exponential change repeatedly")
    print("multiplies. At 8%, the absolute monthly gain grows with the customer base.")
    print("Assumptions and limitations:")
    print("  - The growth rate, churn, and acquisition conditions remain unchanged.")
    print("  - Saturation, competition, capacity, pricing, and costs are not modeled.")
    print("  - Mathematical consistency does not guarantee business realism.")
    print(f"Figures saved to: {growth_path} and {multiplication_path}")
    return 0
