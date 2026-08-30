"""The small, complete analytics experiment used in Chapter 0."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_00_data


def calculate_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the five introductory business metrics."""

    total_revenue = float(data["revenue"].sum())
    total_customers = int(data["customers"].sum())
    return {
        "total_revenue": total_revenue,
        "average_daily_revenue": float(data["revenue"].mean()),
        "total_customers": total_customers,
        "average_revenue_per_customer": total_revenue / total_customers,
        "revenue_per_labor_hour": total_revenue / float(data["labor_hours"].sum()),
    }


def create_revenue_figure(data: pd.DataFrame, output_path: Path) -> Path:
    """Plot daily revenue and save it at ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data["date"], data["revenue"], marker="o")
    axis.set(
        title="Riverside Cafe revenue by date", xlabel="Date", ylabel="Revenue ($)"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run(output_dir: Path | None = None) -> int:
    """Run and narrate the Chapter 0 experiment."""

    data = load_chapter_00_data()
    metrics = calculate_metrics(data)
    grouped = data.groupby("day_of_week", sort=False).agg(
        days_observed=("date", "count"),
        average_revenue=("revenue", "mean"),
        average_customers=("customers", "mean"),
    )
    grouped["revenue_per_labor_hour"] = (
        data.groupby("day_of_week", sort=False)["revenue"].sum()
        / data.groupby("day_of_week", sort=False)["labor_hours"].sum()
    )

    print("Chapter 0 — The Analytics Laboratory")
    print(
        "Question: Which days appear strongest, and what evidence supports that conclusion?\n"
    )
    print("First 5 observations:")
    print(data.head().to_string(index=False, formatters={"revenue": "${:,.2f}".format}))
    print("\nCore metrics:")
    print(f"  Total revenue:                 ${metrics['total_revenue']:,.2f}")
    print(f"  Average daily revenue:         ${metrics['average_daily_revenue']:,.2f}")
    print(f"  Total customers:               {metrics['total_customers']:,}")
    print(
        f"  Average revenue per customer:  ${metrics['average_revenue_per_customer']:,.2f}"
    )
    print(f"  Revenue per labor hour:        ${metrics['revenue_per_labor_hour']:,.2f}")
    print("\nComparison by day of week:")
    print(grouped.round(2).to_string())

    figure_path = (
        output_dir or PROJECT_ROOT / "figures"
    ) / "chapter-00-revenue-by-date.png"
    create_revenue_figure(data, figure_path)
    strongest = grouped["average_revenue"].idxmax()
    print(
        f"\nCalculation: {strongest} had the highest average revenue in these observations."
    )
    print(
        "Interpretation: Weekend demand may be stronger, but 14 days cannot show "
        "that the day itself caused higher revenue. Promotions, weather, or other "
        "conditions may help explain the pattern."
    )
    print("Next question: Are weekends consistently stronger over a longer period?")
    print(f"Figure saved to: {figure_path}")
    return 0
