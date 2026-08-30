"""Matrices as structured business data for Chapter 7."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_07_data


FEATURE_COLUMNS = ["visits", "average_order_value", "months_as_customer"]
SCORE_WEIGHTS = np.array([2.0, 0.1, 0.5])
SECOND_WEIGHTS = np.array([1.5, 0.25, 0.1])


def dataframe_to_matrix(
    df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS
) -> np.ndarray:
    """Select ordered numerical features while leaving labels in the DataFrame."""
    matrix = df.loc[:, columns].to_numpy(dtype=float)
    if matrix.ndim != 2:
        raise ValueError("selected data must form a two-dimensional matrix")
    return matrix


def matrices_compatible(left, right) -> bool:
    """Return whether the inner dimensions permit matrix multiplication."""
    left, right = np.asarray(left), np.asarray(right)
    return left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[0]


def matrix_add(left, right) -> np.ndarray:
    """Add same-shaped matrices entry by entry."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("matrix addition requires equal two-dimensional shapes")
    return left + right


def scalar_multiply(scalar: float, matrix) -> np.ndarray:
    """Multiply every matrix entry by one scalar."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return scalar * matrix


def weighted_scores(matrix, weights=SCORE_WEIGHTS) -> np.ndarray:
    """Apply one feature-weight vector to every observation row."""
    matrix, weights = np.asarray(matrix, dtype=float), np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or weights.ndim != 1 or matrix.shape[1] != weights.shape[0]:
        raise ValueError("matrix columns must match the number of weights")
    return matrix @ weights


def multiple_scores(matrix, weight_matrix) -> np.ndarray:
    """Apply each column of a weight matrix as a separate scoring rule."""
    matrix = np.asarray(matrix, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if not matrices_compatible(matrix, weight_matrix):
        raise ValueError("inside dimensions must match for matrix multiplication")
    return np.matmul(matrix, weight_matrix)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_table_matrix_figure(df: pd.DataFrame, path: Path) -> Path:
    """Visually connect labeled table rows and columns to a numeric matrix."""
    preview = df.loc[:, ["customer_id", *FEATURE_COLUMNS]].head(6)
    matrix = dataframe_to_matrix(preview)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.axis("off")
    left = axes[0].table(
        cellText=preview.values, colLabels=preview.columns, loc="center", cellLoc="center"
    )
    right = axes[1].table(
        cellText=matrix.astype(int), colLabels=FEATURE_COLUMNS, loc="center", cellLoc="center"
    )
    for table in (left, right):
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
    axes[0].set_title("DataFrame: labels retain business meaning")
    axes[1].set_title("Matrix: rows = customers, columns = features")
    fig.suptitle("A business table becomes a numerical matrix")
    return _save(fig, path)


def create_weighted_score_figure(
    customer_ids, scores, path: Path
) -> Path:
    """Plot the common weighted rule applied to every customer row."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(customer_ids, scores, color="tab:blue")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(
        title=r"One weight vector applied to every row: $X\mathbf{w}$",
        xlabel="Customer",
        ylabel="Illustrative weighted score",
    )
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 7 matrix experiment."""
    df = load_chapter_07_data()
    matrix = dataframe_to_matrix(df)
    scores = weighted_scores(matrix)
    weight_matrix = np.column_stack([SCORE_WEIGHTS, SECOND_WEIGHTS])
    outputs = multiple_scores(matrix, weight_matrix)

    manual = 4 * 2 + 25 * 0.1 + 12 * 0.5
    small = matrix[:2]
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    incompatible = (matrix.shape, (2, 2))

    print("Chapter 7 — Matrices: Data Becomes Structure")
    print("\nCustomer table (labels preserve meaning):\n" + df.head().to_string(index=False))
    print(f"\nFeature matrix: shape={matrix.shape}, ndim={matrix.ndim}")
    print(f"First row (Customer A): {matrix[0]}; visits column: {matrix[:, 0]}")
    print(f"x_23 = {matrix[1, 2]:.0f}: Customer B's months as customer.")
    print(f"Transpose example: {small.shape} becomes {small.T.shape}\n{small.T}")
    print(f"Customer A manually: 4(2) + 25(0.1) + 12(0.5) = {manual:.1f}")
    print(f"All scores X @ w: {scores}")
    print(f"Two scores X @ W have shape {outputs.shape}:\n{outputs}")
    print(f"Element-wise A * B:\n{a * b}\nMatrix product A @ B:\n{a @ b}")
    print(
        f"Incompatible example: {incompatible[0]} @ {incompatible[1]} fails because "
        f"inside dimensions {matrix.shape[1]} and 2 do not match."
    )

    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [
        create_table_matrix_figure(df, destination / "chapter-07-table-to-matrix.png"),
        create_weighted_score_figure(
            df.customer_id, scores, destination / "chapter-07-weighted-scores.png"
        ),
    ]
    print("\nInterpretation: each score is one row-by-weight-vector dot product; X @ w performs every row at once.")
    print("Limitation: these invented weights demonstrate mathematics, not a validated customer-value model. Real weights require business judgment, estimation, optimization, statistical modeling, or machine learning.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
