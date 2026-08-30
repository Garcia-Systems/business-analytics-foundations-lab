"""Labeled transaction analysis with pandas for Chapter 10."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_10_data


def prepare_transactions(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Parse dates and derive vectorized revenue and calendar columns."""
    result = (load_chapter_10_data() if df is None else df).copy()
    result["date"] = pd.to_datetime(result["date"])
    result["gross_revenue"] = result["quantity"] * result["unit_price"]
    result["net_revenue"] = result["gross_revenue"] * (1 - result["discount"])
    result["day_of_week"] = result["date"].dt.day_name()
    return result


def high_value_downtown(df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
    """Select Downtown transactions above a net-revenue threshold, largest first."""
    mask = (df["location"] == "Downtown") & (df["net_revenue"] > threshold)
    return df.loc[mask].sort_values("net_revenue", ascending=False)


def category_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per category with total net revenue."""
    return (df.groupby("category", as_index=False)
            .agg(revenue=("net_revenue", "sum"))
            .sort_values("revenue", ascending=False))


def location_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return count, units, total, and mean revenue at location grain."""
    return (df.groupby("location")
            .agg(transactions=("transaction_id", "count"),
                 units=("quantity", "sum"),
                 revenue=("net_revenue", "sum"),
                 average_transaction=("net_revenue", "mean"))
            .sort_values("revenue", ascending=False))


def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Transform transaction-grain rows into one row per date."""
    return df.groupby("date", as_index=False).agg(revenue=("net_revenue", "sum"))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_category_figure(df: pd.DataFrame, path: Path) -> Path:
    summary = category_totals(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary["category"], summary["revenue"], color="tab:blue")
    ax.set(title="Net revenue by category", xlabel="category", ylabel="net revenue ($)")
    ax.grid(axis="y", alpha=.2)
    return _save(fig, path)


def create_location_figure(df: pd.DataFrame, path: Path) -> Path:
    summary = location_metrics(df).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary["location"], summary["revenue"], color="tab:green")
    ax.set(title="Net revenue by location", xlabel="location", ylabel="net revenue ($)")
    ax.grid(axis="y", alpha=.2)
    return _save(fig, path)


def create_daily_figure(df: pd.DataFrame, path: Path) -> Path:
    summary = daily_revenue(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["date"], summary["revenue"], marker="o", color="tab:orange")
    ax.set(title="Daily net revenue", xlabel="date", ylabel="net revenue ($)")
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 10 restaurant DataFrame experiment."""
    raw = load_chapter_10_data()
    df = prepare_transactions(raw)
    selected = high_value_downtown(df)
    categories = category_totals(df)
    locations = location_metrics(df)
    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [
        create_category_figure(df, destination / "chapter-10-revenue-by-category.png"),
        create_location_figure(df, destination / "chapter-10-revenue-by-location.png"),
        create_daily_figure(df, destination / "chapter-10-daily-revenue.png"),
    ]
    top_category = categories.iloc[0]
    top_average = locations["average_transaction"].idxmax()

    print("Chapter 10 — Tables & DataFrames")
    print(f"Loaded shape: {raw.shape}; columns: {', '.join(raw.columns)}")
    print("Dtypes after date parsing: " + ", ".join(f"{c}={t}" for c, t in df.dtypes.items()))
    print("Preview (transaction grain — one row is one transaction):")
    print(df.head(3).to_string(index=False))
    print(f"Downtown net revenue > $50: {len(selected)} transactions; largest={selected.iloc[0]['transaction_id']}")
    print(f"Overall: units={df['quantity'].sum()}, net revenue=${df['net_revenue'].sum():,.2f}, average=${df['net_revenue'].mean():,.2f}")
    print("Revenue by category: " + ", ".join(f"{r.category}=${r.revenue:,.2f}" for r in categories.itertuples()))
    print("Location metrics:\n" + locations.round(2).to_string())
    print(f"Highest-revenue category: {top_category['category']} (${top_category['revenue']:,.2f})")
    print(f"Highest-average-transaction location: {top_average} (${locations.loc[top_average, 'average_transaction']:,.2f})")
    print("Interpretation: totals and averages use different denominators; neither alone defines performance. Ask what one row represents before interpreting a metric.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
