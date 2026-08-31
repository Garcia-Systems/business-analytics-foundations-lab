"""Paired observations, covariance, correlation, and dependence for Chapter 19."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from analytics_foundations.datasets import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chapter-19-location-day.csv"
NUMERIC_FIELDS = ["customers", "revenue", "labor_hours", "temperature"]
REQUIRED_FIGURES = (
    "chapter-19-paired-deviations.png",
    "chapter-19-relationships.png",
    "chapter-19-correlation-matrix.png",
    "chapter-19-outlier-influence.png",
    "chapter-19-nonlinear-dependence.png",
)


def _paired(x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    """Return finite, one-dimensional paired arrays with at least two rows."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("paired values must be one-dimensional")
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    if len(a) < 2:
        raise ValueError("at least two paired observations are required")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired observations must be finite")
    return a, b


def sample_covariance(x: ArrayLike, y: ArrayLike) -> float:
    """Calculate sample covariance directly from deviation products (n - 1)."""
    a, b = _paired(x, y)
    return float(np.sum((a - a.mean()) * (b - b.mean())) / (len(a) - 1))


def sample_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Calculate Pearson sample correlation, rejecting constant variables."""
    a, b = _paired(x, y)
    sx, sy = a.std(ddof=1), b.std(ddof=1)
    if sx == 0 or sy == 0:
        raise ValueError("correlation is undefined when a variable has zero variability")
    return float(sample_covariance(a, b) / (sx * sy))


def variance_of_sum(var_x: float, var_y: float, covariance: float) -> float:
    """Return Var(X + Y) from two variances and their covariance."""
    values = np.asarray([var_x, var_y, covariance], dtype=float)
    if not np.isfinite(values).all() or var_x < 0 or var_y < 0:
        raise ValueError("variances must be finite and nonnegative; covariance must be finite")
    return float(var_x + var_y + 2 * covariance)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the clean one-row-per-location-day dataset."""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "location_id", "customers", "revenue", "labor_hours", "temperature", "promotion_active"}
    if not required.issubset(frame.columns) or frame.duplicated(["date", "location_id"]).any():
        raise ValueError("dataset must contain required fields at unique location-day grain")
    return frame


def outlier_example() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return a deterministic weak cloud and that same cloud plus one influential point."""
    x = np.arange(1., 11.)
    y = np.array([5, 7, 4, 6, 5, 4, 7, 5, 6, 4], dtype=float)
    return x, y, np.append(x, 25), np.append(y, 25)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create the chapter's paired-data diagnostic and teaching figures."""
    output_dir = Path(output_dir); paths: list[Path] = []
    x, y = df.customers.to_numpy(), df.revenue.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(x, y, alpha=.7)
    ax.axvline(x.mean(), color="#e45756", ls="--", label="mean customers")
    ax.axhline(y.mean(), color="#72b7b2", ls="--", label="mean revenue")
    ax.set(title="Paired deviations: same-side quadrants contribute positively", xlabel="Customers", ylabel="Revenue ($)"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["customers", "labor_hours", "temperature"], ["Clear positive", "Positive—not causal", "Weak linear relationship"], strict=True):
        ax.scatter(df[col], y, alpha=.65); ax.set(title=title, xlabel=col.replace("_", " "), ylabel="Revenue ($)")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    corr = df[NUMERIC_FIELDS].corr()
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=NUMERIC_FIELDS, yticklabels=NUMERIC_FIELDS, title="Pearson correlation matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="r"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    xo, yo, xw, yw = outlier_example(); fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, a, b, label in zip(axes, [xo, xw], [yo, yw], ["Without influential point", "With influential point"], strict=True):
        ax.scatter(a, b); ax.set(title=f"{label}\nr = {sample_correlation(a, b):.2f}", xlabel="X", ylabel="Y")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))

    nx = np.arange(-3., 4.); ny = nx**2; fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(nx, ny, s=60)
    ax.plot(nx, ny, alpha=.5); ax.set(title=f"Deterministic U-shape, Pearson r = {sample_correlation(nx, ny):.2f}", xlabel="X", ylabel="Y = X²")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4])); return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 19 location-day dependence experiment."""
    df = load_data(); x, y = df.customers.to_numpy(), df.revenue.to_numpy()
    dx, dy = x - x.mean(), y - y.mean(); products = dx * dy
    manual, numpy_cov, pandas_cov = sample_covariance(x, y), np.cov(x, y, ddof=1)[0, 1], df[["customers", "revenue"]].cov().iloc[0, 1]
    corr = sample_correlation(x, y); cents = 100 * y
    cov_matrix, corr_matrix = df[NUMERIC_FIELDS].cov(), df[NUMERIC_FIELDS].corr()
    xo, yo, xw, yw = outlier_example(); nx = np.arange(-3., 4.); ny = nx**2
    print("Chapter 19 — Covariance & Dependence")
    print(f"Dataset: {len(df)} rows; grain = one unique location-day. Main pair: customers + revenue; interpretation pair: labor hours + revenue; weak pair: temperature + revenue.")
    print(f"Means: customers={x.mean():.2f}, revenue=${y.mean():,.2f}. First four (x deviation, y deviation, product):")
    for a, b, product in zip(dx[:4], dy[:4], products[:4], strict=True): print(f"  ({a:+.2f}, {b:+.2f}) -> {product:+,.2f}")
    print(f"Sample covariance (manual / NumPy / pandas): {manual:,.2f} / {numpy_cov:,.2f} / {pandas_cov:,.2f}; Pearson r={corr:.4f}.")
    print(f"Dollars→cents: covariance ×{sample_covariance(x, cents)/manual:.0f}; correlation {sample_correlation(x, cents):.4f} (unchanged).")
    print("Covariance matrix (diagonal=variances, off-diagonal=covariances; symmetric):\n" + cov_matrix.round(2).to_string())
    print("Correlation matrix (unitless, symmetric, diagonal=1):\n" + corr_matrix.round(3).to_string())
    print(f"Outlier influence: r changes from {sample_correlation(xo, yo):.3f} to {sample_correlation(xw, yw):.3f}.")
    print(f"Nonlinear dependence: Y=X² determines Y exactly, but Pearson r={sample_correlation(nx, ny):.3f}; zero correlation is not independence.")
    print(f"Variance of sum: 100+225+2(50)={variance_of_sum(100, 225, 50):.0f}; at covariance 0: 325; at -50: {variance_of_sum(100, 225, -50):.0f}.")
    paths = create_figures(df, output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures. Positive labor–revenue correlation does NOT show that adding labor causes revenue: expected demand, weekends, promotions, reservations, size, season, or events may drive both.")
    return 0
