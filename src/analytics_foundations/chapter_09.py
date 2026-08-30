"""NumPy arrays and vectorized restaurant analysis for Chapter 9."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytics_foundations.datasets import PROJECT_ROOT


LOCATIONS = np.array(["Downtown", "Riverside", "Midtown", "Campus", "Harbor"])
DAYS = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
REVENUE = np.array([
    [4200, 4600, 4400, 5100, 6200, 7000, 5900],
    [3800, 4000, 4150, 4700, 5200, 6100, 5400],
    [4500, 4800, 5000, 5400, 6500, 7200, 6300],
    [3500, 3900, 4100, 4600, 5700, 6400, 5200],
    [4100, 4300, 4550, 5000, 6000, 6800, 5800],
])
CUSTOMERS = np.array([
    [140, 150, 145, 165, 195, 220, 185],
    [125, 132, 136, 150, 170, 198, 176],
    [145, 155, 162, 174, 205, 225, 198],
    [118, 128, 135, 148, 180, 205, 170],
    [135, 142, 150, 162, 190, 214, 181],
])
LABOR_HOURS = np.array([
    [78, 80, 79, 86, 94, 102, 92],
    [72, 74, 76, 82, 89, 98, 90],
    [80, 83, 86, 91, 99, 106, 98],
    [68, 72, 75, 80, 90, 99, 87],
    [75, 77, 80, 85, 93, 101, 91],
], dtype=float)
TARGETS = np.array([5000, 4800, 5200, 4500, 5100])


def business_metrics(revenue=REVENUE, customers=CUSTOMERS,
                     labor_hours=LABOR_HOURS) -> dict[str, np.ndarray]:
    """Calculate normalized metrics once over every location-day observation."""
    revenue = np.asarray(revenue, dtype=float)
    customers = np.asarray(customers, dtype=float)
    labor_hours = np.asarray(labor_hours, dtype=float)
    if revenue.shape != customers.shape or revenue.shape != labor_hours.shape:
        raise ValueError("revenue, customers, and labor_hours must share a shape")
    return {
        "revenue_per_customer": revenue / customers,
        "revenue_per_labor_hour": revenue / labor_hours,
    }


def target_deviations(revenue=REVENUE, targets=TARGETS) -> np.ndarray:
    """Broadcast one daily target per location across all days."""
    revenue = np.asarray(revenue)
    targets = np.asarray(targets)
    if revenue.ndim != 2 or targets.shape != (revenue.shape[0],):
        raise ValueError("targets must contain one value per revenue row")
    return revenue - targets[:, np.newaxis]


def standardize(x) -> np.ndarray:
    """Return a sample-standardized copy of a one-dimensional array."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x must contain at least two values in one dimension")
    return (values - values.mean()) / values.std(ddof=1)


def safe_slice(x, start: int, stop: int) -> np.ndarray:
    """Return an independently modifiable copy of a slice."""
    return np.asarray(x)[start:stop].copy()


def summarize(revenue=REVENUE) -> dict[str, object]:
    """Aggregate a location-by-day matrix and locate its strongest extremes."""
    revenue = np.asarray(revenue)
    if revenue.ndim != 2:
        raise ValueError("revenue must be a two-dimensional array")
    by_location = revenue.sum(axis=1)
    by_day = revenue.sum(axis=0)
    return {
        "total": revenue.sum(),
        "by_location": by_location,
        "by_day": by_day,
        "average_by_location": revenue.mean(axis=1),
        "best_location": int(np.argmax(by_location)),
        "worst_location": int(np.argmin(by_location)),
        "best_day": int(np.argmax(by_day)),
        "worst_day": int(np.argmin(by_day)),
    }


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_revenue_heatmap(revenue, path: Path) -> Path:
    """Render the two-dimensional revenue array as a labeled image."""
    revenue = np.asarray(revenue)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(revenue, cmap="YlGn", aspect="auto")
    ax.set(xticks=np.arange(len(DAYS)), xticklabels=DAYS,
           yticks=np.arange(len(LOCATIONS)), yticklabels=LOCATIONS,
           xlabel="day (column)", ylabel="location (row)",
           title="Daily revenue matrix")
    for row, column in np.ndindex(revenue.shape):
        ax.text(column, row, f"${revenue[row, column] / 1000:.1f}k",
                ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="revenue ($)")
    return _save(fig, path)


def create_location_totals_figure(revenue, path: Path) -> Path:
    """Plot the result of reducing the day dimension with sum(axis=1)."""
    totals = np.asarray(revenue).sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(LOCATIONS, totals, color="tab:blue")
    ax.set(title="Revenue by location: (5, 7) → (5,)", ylabel="weekly revenue ($)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.2)
    return _save(fig, path)


def create_target_deviation_figure(revenue, targets, path: Path) -> Path:
    """Plot broadcast daily target deviations for every location."""
    deviations = target_deviations(revenue, targets)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, location in enumerate(LOCATIONS):
        ax.plot(DAYS, deviations[index], marker="o", label=location)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Daily revenue minus each location's target",
           xlabel="day", ylabel="target deviation ($)")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 9 restaurant-array experiment."""
    metrics = business_metrics()
    summary = summarize()
    deviations = target_deviations()
    mask = (REVENUE > 5000) & (LABOR_HOURS < 100)
    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [
        create_revenue_heatmap(REVENUE, destination / "chapter-09-revenue-matrix.png"),
        create_location_totals_figure(REVENUE, destination / "chapter-09-location-totals.png"),
        create_target_deviation_figure(REVENUE, TARGETS, destination / "chapter-09-target-deviation.png"),
    ]

    raw = [100, 120, 90]
    array = np.array(raw)
    print("Chapter 9 — Arrays & Vectorized Thinking")
    print(f"List × 2 repeats: {raw * 2}; array × 2 computes: {array * 2}")
    print(f"Revenue shape={REVENUE.shape}, ndim={REVENUE.ndim}, size={REVENUE.size}, dtype={REVENUE.dtype}")
    print(f"First location, first three days: {REVENUE[0, :3]}")
    print(f"Revenue/customer mean=${metrics['revenue_per_customer'].mean():.2f}; revenue/labor-hour mean=${metrics['revenue_per_labor_hour'].mean():.2f}")
    print(f"High-revenue, under-100-hour observations: {REVENUE[mask]} ({mask.sum()} matches)")
    print(f"All revenue=${summary['total']:,}; by location={summary['by_location']}; by day={summary['by_day']}")
    print(f"Shape reasoning: sum(axis=0) {REVENUE.shape} → {summary['by_day'].shape}; sum(axis=1) {REVENUE.shape} → {summary['by_location'].shape}")
    print(f"Best location={LOCATIONS[summary['best_location']]}; best day={DAYS[summary['best_day']]}; worst day={DAYS[summary['worst_day']]}")
    print(f"Broadcasting: revenue {REVENUE.shape} - targets[:, newaxis] {TARGETS[:, np.newaxis].shape} = deviations {deviations.shape}")
    print(f"Target classifications: above={np.count_nonzero(np.where(deviations >= 0, True, False))}, below={np.count_nonzero(deviations < 0)}")
    print(f"Reshape: targets {TARGETS.shape} → column {TARGETS.reshape(-1, 1).shape}; standardized location totals={np.round(standardize(summary['by_location']), 2)}")
    print("Interpretation: Midtown has the largest weekly total, but totals do not establish efficiency; compare per-customer, per-labor-hour, and margin metrics before deciding.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
