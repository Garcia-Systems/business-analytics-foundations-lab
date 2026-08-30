"""Vectors, distance, and weighted combinations for Chapter 6."""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_06_data


FEATURES = ["visits", "average_order_value"]


def _pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Return compatible one-dimensional numeric vectors."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("vectors must be one-dimensional and have equal dimensions")
    return x, y


def vector_add(x, y) -> np.ndarray:
    """Add corresponding vector components."""
    x, y = _pair(x, y)
    return x + y


def vector_subtract(x, y) -> np.ndarray:
    """Subtract corresponding components, returning displacement from y to x."""
    x, y = _pair(x, y)
    return x - y


def scalar_multiply(c: float, x) -> np.ndarray:
    """Multiply every component of a vector by scalar c."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional vector")
    return c * x


def magnitude(x) -> float:
    """Return Euclidean vector length, calculated from squared components."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional vector")
    return float(np.sqrt(np.sum(x**2)))


def euclidean_distance(x, y) -> float:
    """Return the magnitude of the displacement between two observations."""
    return magnitude(vector_subtract(x, y))


def dot_product(x, y) -> float:
    """Multiply corresponding components and sum."""
    x, y = _pair(x, y)
    return float(np.sum(x * y))


def row_to_vector(row: pd.Series, features=FEATURES) -> np.ndarray:
    """Convert selected numeric fields in a DataFrame row to a feature vector."""
    return row.loc[list(features)].to_numpy(dtype=float)


def closest_pair(features: pd.DataFrame) -> tuple[str, str, float]:
    """Find the closest row pair by raw Euclidean distance (not clustering)."""
    best: tuple[str, str, float] | None = None
    for i, j in combinations(range(len(features)), 2):
        distance = euclidean_distance(features.iloc[i], features.iloc[j])
        candidate = (str(features.index[i]), str(features.index[j]), distance)
        if best is None or distance < best[2]:
            best = candidate
    if best is None:
        raise ValueError("at least two observations are required")
    return best


def feature_scale_example() -> dict[str, float | str]:
    """Show that rescaling annual spend can reverse the nearest neighbor."""
    anchor = np.array([2.0, 100.0])
    frequent = np.array([9.0, 110.0])
    spend_alike = np.array([3.0, 500.0])
    raw_b = euclidean_distance(anchor, frequent)
    raw_c = euclidean_distance(anchor, spend_alike)
    scale = np.array([1.0, 1 / 1000])
    scaled_b = euclidean_distance(anchor * scale, frequent * scale)
    scaled_c = euclidean_distance(anchor * scale, spend_alike * scale)
    return {"raw_frequent": raw_b, "raw_spend_alike": raw_c,
            "scaled_frequent": scaled_b, "scaled_spend_alike": scaled_c,
            "raw_nearest": "frequent", "scaled_nearest": "spend_alike"}


def weighted_revenue(quantities=(10, 4, 2), prices=(5, 8, 12)) -> float:
    """Compute total revenue as quantity dot price."""
    return dot_product(quantities, prices)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_customer_points_figure(df: pd.DataFrame, path: Path) -> Path:
    """Plot each table row as a point and Customer A as an origin vector."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df.visits, df.average_order_value)
    for row in df.itertuples(): ax.annotate(row.customer_id, (row.visits, row.average_order_value), xytext=(4, 4), textcoords="offset points")
    a = df.iloc[0]
    ax.annotate("", xy=(a.visits, a.average_order_value), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2})
    ax.set(title="Cafe customers become points (and vectors)", xlabel="Visits", ylabel="Average order value ($)")
    ax.set_xlim(0, 11); ax.set_ylim(0, 50); ax.grid(alpha=.2)
    return _save(fig, path)


def create_distance_figure(df: pd.DataFrame, path: Path, first="A", second="B") -> Path:
    """Draw the displacement segment and its Euclidean length."""
    indexed = df.set_index("customer_id"); a, b = indexed.loc[first], indexed.loc[second]
    distance = euclidean_distance(row_to_vector(a), row_to_vector(b))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(df.visits, df.average_order_value, color="lightgray")
    ax.plot([a.visits, b.visits], [a.average_order_value, b.average_order_value], "o-", color="tab:red", lw=2, label=f"distance = {distance:.2f}")
    ax.annotate(first, (a.visits, a.average_order_value)); ax.annotate(second, (b.visits, b.average_order_value))
    ax.set(title=f"Subtraction gives displacement from {first} to {second}", xlabel="Visits", ylabel="Average order value ($)"); ax.grid(alpha=.2); ax.legend()
    return _save(fig, path)


def create_scale_figure(path: Path) -> Path:
    """Compare geometry before and after expressing spend in thousands."""
    points = {"Anchor": (2, 100), "Frequent": (9, 110), "Spend-alike": (3, 500)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, divisor, title in zip(axes, (1, 1000), ("Raw dollars", "Spend in $1,000s"), strict=True):
        for name, (visits, spend) in points.items(): ax.scatter(visits, spend/divisor); ax.annotate(name, (visits, spend/divisor))
        ax.set(title=title, xlabel="Visits", ylabel="Annual spend" + (" ($)" if divisor == 1 else " ($000s)")); ax.grid(alpha=.2)
    fig.suptitle("Changing feature scale changes apparent distance")
    return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 6 customer-vector experiment."""
    df = load_chapter_06_data(); features = df[FEATURES].copy(); features.index = df.customer_id
    a, b = row_to_vector(features.loc["A"]), row_to_vector(features.loc["B"])
    difference = vector_subtract(b, a); distance = euclidean_distance(a, b)
    pair = closest_pair(features); scale = feature_scale_example()
    print("Chapter 6 — Vectors: Data Becomes Geometry")
    print("Scenario: ten fictional Harbor Cafe customers represented by visits, average order value, and tenure.")
    print("\nCustomer preview:\n" + df.head().to_string(index=False))
    print(f"\nA = {a}; B = {b}; B−A = {difference}.")
    print(f"Magnitudes: ||A||={magnitude(a):.2f}, ||B||={magnitude(b):.2f}; distance ||B−A||={distance:.2f}.")
    print(f"Closest raw-feature pair: {pair[0]} and {pair[1]} (distance {pair[2]:.2f}).")
    print(f"Scale example: raw nearest={scale['raw_nearest']}; after spend is expressed in $1,000s, nearest={scale['scaled_nearest']}.")
    x, y = np.array([2., 3.]), np.array([4., 1.])
    print(f"Dot product: {x}·{y} = {dot_product(x, y):.0f}; element-wise product = {x*y}.")
    print(f"Weighted revenue: [10, 4, 2] @ [5, 8, 12] = ${weighted_revenue():.0f}.")
    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [create_customer_points_figure(df, destination / "chapter-06-customer-points.png"),
             create_distance_figure(df, destination / "chapter-06-distance.png"),
             create_scale_figure(destination / "chapter-06-feature-scale.png")]
    print("\nInterpretation: smaller distance means more similar only under the selected features, scale, and Euclidean metric.")
    print("Assumptions/limits: clean numeric features are treated as equally meaningful; omitted tenure, preferences, profitability, geography, and promotion response may change the conclusion. This is comparison, not clustering.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
