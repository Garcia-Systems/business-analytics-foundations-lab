"""Systems, rank, and least-squares models for Chapter 8."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytics_foundations.datasets import PROJECT_ROOT


BUSINESS_A = np.array([[2.0, 1.0], [1.0, 3.0]])
BUSINESS_B = np.array([110.0, 170.0])
DEPENDENT_A = np.array([[1.0, 2.0], [2.0, 4.0]])


def linear_combination(vectors, coefficients) -> np.ndarray:
    """Return the coefficient-weighted combination of matrix columns."""
    vectors = np.asarray(vectors, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if vectors.ndim != 2 or coefficients.ndim != 1:
        raise ValueError("vectors must be a matrix and coefficients a vector")
    if vectors.shape[1] != coefficients.size:
        raise ValueError("one coefficient is required for each column")
    return vectors @ coefficients


def solve_exact_system(A=BUSINESS_A, b=BUSINESS_B) -> np.ndarray:
    """Solve a square, full-rank system without explicitly forming an inverse."""
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))


def classify_system(A, b) -> str:
    """Classify a linear system as unique, dependent, or inconsistent."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    rank_a = np.linalg.matrix_rank(A)
    rank_augmented = np.linalg.matrix_rank(np.column_stack([A, b]))
    if rank_a < rank_augmented:
        return "inconsistent"
    if rank_a < A.shape[1]:
        return "dependent"
    return "unique"


def intercept_matrix(x) -> np.ndarray:
    """Create a model matrix containing an intercept and one feature."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    return np.column_stack([np.ones(x.size), x])


def least_squares_fit(
    X, y
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray]:
    """Fit least squares and return coefficients, predictions, residuals, and diagnostics."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    beta, _reported_residuals, rank, singular_values = np.linalg.lstsq(
        X, y, rcond=None
    )
    predictions = X @ beta
    residuals = y - predictions
    rss = float(residuals @ residuals)
    return beta, predictions, residuals, rss, int(rank), singular_values


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_system_figure(kind: str, path: Path) -> Path:
    """Plot the unique, dependent, or inconsistent two-equation system."""
    x = np.linspace(-1, 8, 300)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if kind == "unique":
        ax.plot(x, 110 - 2 * x, label=r"$2s+p=110$")
        ax.plot(x, (170 - x) / 3, label=r"$s+3p=170$")
        ax.scatter([32], [46], color="black", zorder=3)
        ax.annotate(
            "solution (32, 46)", (32, 46), xytext=(-48, 14),
            textcoords="offset points"
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 120)
        ax.set(
            xlabel="standard contribution s", ylabel="premium contribution p",
            title="One intersection: one solution"
        )
    elif kind in {"dependent", "inconsistent"}:
        ax.plot(x, (6 - x) / 2, linewidth=4, alpha=.65, label=r"$x+2y=6$")
        rhs = 12 if kind == "dependent" else 15
        ax.plot(x, (rhs - 2 * x) / 4, linestyle="--", label=rf"$2x+4y={rhs}$")
        title = (
            "The same line: infinitely many solutions"
            if kind == "dependent" else "Parallel lines: no exact solution"
        )
        ax.set(xlabel="x", ylabel="y", title=title)
    else:
        raise ValueError("kind must be unique, dependent, or inconsistent")
    ax.grid(alpha=.2)
    ax.legend()
    return _save(fig, path)


def create_least_squares_figure(x, y, beta, path: Path) -> Path:
    """Plot noisy observations, their fitted line, and vertical residuals."""
    x, y, beta = np.asarray(x), np.asarray(y), np.asarray(beta)
    predictions = intercept_matrix(x) @ beta
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(x, y, color="tab:blue", label="observed outcomes", zorder=3)
    order = np.argsort(x)
    ax.plot(x[order], predictions[order], color="tab:orange", label="least-squares line")
    ax.vlines(x, predictions, y, color="tab:red", alpha=.7, label="residuals")
    ax.set(title="Many noisy equations approximated by one model", xlabel="business activity", ylabel="outcome")
    ax.grid(alpha=.2)
    ax.legend()
    return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 8 service-package experiment."""
    solution = solve_exact_system()
    verification = BUSINESS_A @ solution
    dependent_b = np.array([6.0, 12.0])
    inconsistent_b = np.array([6.0, 15.0])
    x = np.arange(1.0, 6.0)
    y = np.array([9.0, 12.0, 16.0, 19.0, 23.0])
    X = intercept_matrix(x)
    beta, predictions, residuals, rss, rank, _ = least_squares_fit(X, y)

    print("Chapter 8 — Linear Algebra for Models")
    print("\nService-package contribution system:")
    print("  2s + p = 110\n  s + 3p = 170")
    print(f"A =\n{BUSINESS_A}\nx = [s, p]\nb = {BUSINESS_B}\nA x = b")
    print(f"Direct solution: s = {solution[0]:.2f}, p = {solution[1]:.2f}")
    print(f"Verification A @ x = {verification}")
    print(f"Unique rank: {np.linalg.matrix_rank(BUSINESS_A)} ({classify_system(BUSINESS_A, BUSINESS_B)})")
    print(f"Dependent rank: {np.linalg.matrix_rank(DEPENDENT_A)} ({classify_system(DEPENDENT_A, dependent_b)})")
    print(f"Changed total: {classify_system(DEPENDENT_A, inconsistent_b)}; no exact solution")
    print("\nFive noisy observations, two coefficients:")
    print(f"beta = {np.round(beta, 3)}; rank = {rank}")
    print(f"predictions = {np.round(predictions, 3)}")
    print(f"residuals = {np.round(residuals, 3)}; residual sum of squares = {rss:.3f}")

    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [
        create_system_figure(kind, destination / f"chapter-08-{kind}-system.png")
        for kind in ("unique", "dependent", "inconsistent")
    ]
    paths.append(create_least_squares_figure(x, y, beta, destination / "chapter-08-least-squares.png"))
    print("\nInterpretation: exact systems can reveal fixed contributions; noisy business observations usually require coefficients that best approximate all outcomes.")
    print("Limitations: this simplified model omits measurement noise, changing behavior, prices, promotions, operational variation, and other variables. Small residuals alone do not establish causality, future accuracy, or business value.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
