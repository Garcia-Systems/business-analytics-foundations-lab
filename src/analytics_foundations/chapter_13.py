"""Exploratory, grain-aware visualization for Chapter 13."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_13_data

GRAIN = "location-day"
REQUIRED_FIGURES = (
    "chapter-13-revenue-histogram.png",
    "chapter-13-location-boxplots.png",
    "chapter-13-location-total-bar.png",
    "chapter-13-labor-revenue-scatter.png",
    "chapter-13-raw-vs-aggregate.png",
    "chapter-13-revenue-over-time.png",
    "chapter-13-promotion-boxplots.png",
    "chapter-13-axis-comparison.png",
)


def validate_grain(df: pd.DataFrame) -> None:
    """Require one complete observation per location and date."""
    if df.duplicated(["location_id", "date"]).any():
        raise ValueError("(location_id, date) must uniquely identify each row")
    if df[["location_id", "date", "revenue", "labor_hours"]].isna().any().any():
        raise ValueError("key analytical fields must be complete")


def revenue_summary(df: pd.DataFrame) -> pd.Series:
    """Return numerical evidence used alongside the distribution plot."""
    return df["revenue"].agg(["count", "mean", "median", "min", "max"])


def location_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Create location-grain aggregates, ordered for visual comparison."""
    return (df.groupby("location_name", as_index=False)
            .agg(revenue=("revenue", "sum"), days=("date", "count"))
            .sort_values("revenue"))


def prepare_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across locations and sort before connecting observations."""
    return (df.groupby("date", as_index=False).agg(revenue=("revenue", "sum"))
            .sort_values("date").reset_index(drop=True))


def location_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Change from location-day points to one average point per location."""
    return (df.groupby("location_name", as_index=False)
            .agg(labor_hours=("labor_hours", "mean"), revenue=("revenue", "mean")))


def outlier_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve rows beyond the familiar 1.5-IQR candidate fences."""
    q1, q3 = df["revenue"].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df.loc[(df.revenue < q1 - 1.5 * iqr) | (df.revenue > q3 + 1.5 * iqr)].copy()


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate deterministic Matplotlib evidence without an interactive display."""
    paths: list[Path] = []
    mean, median = df.revenue.mean(), df.revenue.median()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(df.revenue, bins=12, color="#4c78a8", edgecolor="white"); ax.axvline(mean, color="#e45756", label=f"Mean ${mean:,.0f}"); ax.axvline(median, color="#54a24b", linestyle="--", label=f"Median ${median:,.0f}"); ax.set(title="How Is Daily Revenue Distributed?", xlabel="Daily revenue ($)", ylabel="Location-days (count)"); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))
    names=sorted(df.location_name.unique()); values=[df.loc[df.location_name.eq(n),"revenue"] for n in names]
    fig,ax=plt.subplots(figsize=(8,4)); ax.boxplot(values,tick_labels=names); ax.set(title="Distribution of Daily Revenue by Location",xlabel="Location",ylabel="Daily revenue ($)"); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[1]))
    totals=location_totals(df); fig,ax=plt.subplots(figsize=(8,4)); ax.bar(totals.location_name,totals.revenue,color="#72b7b2"); ax.set(title="Total Revenue by Location (Aggregate)",xlabel="Location",ylabel="Total revenue ($)"); ax.set_ylim(0,None); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[2]))
    fig,ax=plt.subplots(figsize=(8,4));
    for name,g in df.groupby("location_name",sort=True): ax.scatter(g.labor_hours,g.revenue,alpha=.4,label=name)
    ax.set(title="Are Higher-Labor Days Associated with Higher Revenue?",xlabel="Labor hours",ylabel="Daily revenue ($)"); ax.legend(); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[3]))
    avg=location_averages(df); fig,(a,b)=plt.subplots(1,2,figsize=(10,4)); a.scatter(df.labor_hours,df.revenue,alpha=.4); a.set(title="Raw: one point = location-day",xlabel="Labor hours",ylabel="Daily revenue ($)"); b.scatter(avg.labor_hours,avg.revenue,s=65); [b.annotate(r.location_name,(r.labor_hours,r.revenue),xytext=(3,3),textcoords="offset points") for r in avg.itertuples()]; b.set(title="Aggregated: one point = location",xlabel="Mean labor hours",ylabel="Mean daily revenue ($)"); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[4]))
    daily=prepare_daily_series(df); fig,ax=plt.subplots(figsize=(9,4)); ax.plot(daily.date,daily.revenue,marker="o"); ax.set(title="Restaurant-Group Revenue over Time",xlabel="Date",ylabel="Daily revenue across locations ($)"); ax.tick_params(axis="x",rotation=30); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[5]))
    groups=[df.loc[~df.promotion_active,"revenue"],df.loc[df.promotion_active,"revenue"]]; fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot(groups,tick_labels=["No promotion","Promotion"]); ax.set(title="Revenue Distribution by Promotion Status",xlabel="Observed day type",ylabel="Daily revenue ($)"); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[6]))
    example=pd.Series({"Location A":100,"Location B":105}); fig,(a,b)=plt.subplots(1,2,figsize=(9,4)); a.bar(example.index,example.values); a.set_ylim(98,106); a.set(title="Misleading: truncated axis",ylabel="Illustrative index"); b.bar(example.index,example.values); b.set_ylim(0,110); b.set(title="Corrected: zero baseline",ylabel="Illustrative index"); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[7]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run question → visual evidence → cautious interpretation."""
    df=load_chapter_13_data(); validate_grain(df); summary=revenue_summary(df); totals=location_totals(df); unusual=outlier_candidates(df); paths=create_figures(df,output_dir or PROJECT_ROOT/"figures")
    print("Chapter 13 — Seeing Data | question → variables → visual form → pattern → challenge → interpret")
    print(f"Dataset: {len(df)} rows; grain = one {GRAIN}; {df.location_id.nunique()} locations × {df.date.nunique()} dates; composite keys unique.")
    print("Revenue summary: "+", ".join(f"{k}={v:,.2f}" for k,v in summary.items()))
    print(f"Location aggregates reconcile={totals.revenue.sum() == df.revenue.sum()}; time data ordered={prepare_daily_series(df).date.is_monotonic_increasing}.")
    if unusual.empty: print("Visual anomaly follow-up: no 1.5-IQR candidates; plots still invite inspection.")
    else: print("Visual anomaly follow-up (table evidence):\n"+unusual[["date","location_name","revenue","labor_hours","promotion_active"]].to_string(index=False))
    promo=df.groupby("promotion_active").revenue.median(); print(f"Observation: sample median revenue is ${promo.get(True,float('nan')):,.0f} on promotion days and ${promo.get(False,float('nan')):,.0f} otherwise.")
    print("Possible explanation: promotions, weekends, location mix, or expected demand may coincide. Unsupported conclusion: the plot does not prove promotions or added labor caused revenue.")
    print(f"Generated {len(paths)} figures, including raw-versus-aggregate grain and clearly labeled truncated-versus-zero-axis evidence.")
    return 0
