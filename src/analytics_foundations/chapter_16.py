"""Random variables, discrete probability models, and simulation for Chapter 16."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from analytics_foundations.datasets import PROJECT_ROOT


VALUES = np.array([80, 120, 160, 200, 240])
PROBABILITIES = np.array([.10, .20, .35, .25, .10])
REQUIRED_FIGURES = (
    "chapter-16-outcome-mapping.png",
    "chapter-16-pmf.png",
    "chapter-16-cdf.png",
    "chapter-16-simulation-versus-model.png",
    "chapter-16-continuous-preview.png",
)


def _arrays(values: ArrayLike, probabilities: ArrayLike) -> tuple[NDArray, NDArray[np.float64]]:
    values_array = np.asarray(values)
    probabilities_array = np.asarray(probabilities, dtype=float)
    if values_array.ndim != 1 or probabilities_array.ndim != 1:
        raise ValueError("values and probabilities must be one-dimensional")
    if len(values_array) != len(probabilities_array):
        raise ValueError("values and probabilities must have matching lengths")
    if len(values_array) == 0:
        raise ValueError("a PMF must contain at least one value")
    return values_array, probabilities_array


def validate_pmf(values: ArrayLike, probabilities: ArrayLike) -> None:
    """Raise ``ValueError`` unless values and probabilities define a valid PMF."""
    values_array, probabilities_array = _arrays(values, probabilities)
    if len(np.unique(values_array)) != len(values_array):
        raise ValueError("PMF values must be unique")
    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities must be finite")
    if np.any(probabilities_array < 0):
        raise ValueError("probabilities cannot be negative")
    if not np.isclose(probabilities_array.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")


def exact_probability(values: ArrayLike, probabilities: ArrayLike, x: float) -> float:
    """Return P(X=x), including zero when x is outside the support."""
    values_array, probabilities_array = _arrays(values, probabilities)
    validate_pmf(values_array, probabilities_array)
    return float(probabilities_array[values_array == x].sum())


def threshold_probability(
    values: ArrayLike, probabilities: ArrayLike, threshold: float, *,
    comparison: str = ">=",
) -> float:
    """Return probability above or below a threshold using a simple operator."""
    values_array, probabilities_array = _arrays(values, probabilities)
    validate_pmf(values_array, probabilities_array)
    operators = {
        ">=": np.greater_equal, ">": np.greater,
        "<=": np.less_equal, "<": np.less,
    }
    if comparison not in operators:
        raise ValueError("comparison must be one of >=, >, <=, or <")
    return float(probabilities_array[operators[comparison](values_array, threshold)].sum())


def probability_between(
    values: ArrayLike, probabilities: ArrayLike, lower: float, upper: float,
) -> float:
    """Return P(lower <= X <= upper) for a discrete PMF."""
    if lower > upper:
        raise ValueError("lower cannot exceed upper")
    values_array, probabilities_array = _arrays(values, probabilities)
    validate_pmf(values_array, probabilities_array)
    mask = (values_array >= lower) & (values_array <= upper)
    return float(probabilities_array[mask].sum())


def discrete_cdf(values: ArrayLike, probabilities: ArrayLike, x: float) -> float:
    """Return F_X(x)=P(X<=x) for a discrete PMF."""
    return threshold_probability(values, probabilities, x, comparison="<=")


def indicator(values: ArrayLike, threshold: float = 200) -> NDArray[np.int64]:
    """Map support or observed values to 1 at/above a business threshold, else 0."""
    return (np.asarray(values) >= threshold).astype(np.int64)


def revenue_transform(values: ArrayLike, revenue_per_customer: float = 18) -> NDArray:
    """Transform customer counts using R=revenue_per_customer*X."""
    return np.asarray(values) * revenue_per_customer


def simulate_discrete(
    values: ArrayLike, probabilities: ArrayLike, size: int = 1_000, *, seed: int = 16,
) -> NDArray:
    """Draw reproducible realizations from a validated discrete PMF."""
    values_array, probabilities_array = _arrays(values, probabilities)
    validate_pmf(values_array, probabilities_array)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive integer")
    return np.random.default_rng(seed).choice(values_array, size=size, p=probabilities_array)


def empirical_proportions(samples: ArrayLike, support: ArrayLike) -> NDArray[np.float64]:
    """Return sample proportions aligned to the supplied support."""
    samples_array = np.asarray(samples)
    support_array = np.asarray(support)
    if samples_array.size == 0:
        raise ValueError("samples cannot be empty")
    return np.array([(samples_array == value).mean() for value in support_array])


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_figures(output_dir: Path, *, seed: int = 16) -> list[Path]:
    """Create mapping, PMF, CDF, simulation, and continuous-preview figures."""
    output_dir = Path(output_dir)
    paths: list[Path] = []
    outcomes = ["Low", "Moderate", "Busy", "Very Busy"]
    mapped = [80, 120, 180, 240]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.axis("off")
    for row, (outcome, value) in enumerate(zip(outcomes, mapped, strict=True)):
        y = .85 - row * .22
        ax.text(.24, y, outcome, ha="right", va="center", fontsize=13)
        ax.annotate("", xy=(.7, y), xytext=(.34, y), arrowprops={"arrowstyle": "->", "lw": 2})
        ax.text(.76, y, str(value), ha="left", va="center", fontsize=13)
    ax.set_title("A random variable maps uncertain outcomes to numbers")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markerline, stemlines, baseline = ax.stem(VALUES, PROBABILITIES)
    plt.setp(stemlines, linewidth=3, color="#4c78a8"); plt.setp(markerline, markersize=8)
    ax.set(xticks=VALUES, xlabel="Customer count x", ylabel="P(X=x)", title="Discrete customer-count PMF", ylim=(0, .4))
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    cumulative = np.cumsum(PROBABILITIES)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_x = np.r_[VALUES[0] - 40, VALUES, VALUES[-1] + 40]
    plot_y = np.r_[0, cumulative, 1]
    ax.step(plot_x, plot_y, where="post", color="#f28e2b", linewidth=2.5)
    ax.scatter(VALUES, cumulative, color="#f28e2b", zorder=3)
    ax.set(xticks=VALUES, xlabel="Customer count x", ylabel="F_X(x)=P(X≤x)", title="Probability accumulates in jumps", ylim=(-.03, 1.06))
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    samples = simulate_discrete(VALUES, PROBABILITIES, seed=seed)
    empirical = empirical_proportions(samples, VALUES)
    fig, ax = plt.subplots(figsize=(8, 4.5)); width = 14
    ax.bar(VALUES-width/2, PROBABILITIES, width=width, label="Model PMF")
    ax.bar(VALUES+width/2, empirical, width=width, label="Simulated sample", color="#f28e2b")
    ax.set(xticks=VALUES, xlabel="Customer count", ylabel="Proportion", title="Model probabilities versus empirical proportions")
    ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))

    y_values = np.linspace(5, 15, 400)
    curve = np.exp(-.5 * ((y_values-10)/1.8)**2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(y_values, curve, color="#4c78a8", linewidth=2)
    mask = (y_values >= 9) & (y_values <= 11)
    ax.fill_between(y_values[mask], curve[mask], color="#72b7b2", alpha=.65)
    ax.set(xlabel="Order preparation time Y (minutes)", ylabel="Conceptual density", title="Continuous preview: probability corresponds to area over an interval")
    ax.text(10, .35, "P(9≤Y≤11)", ha="center")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 16 random-variable experiment."""
    validate_pmf(VALUES, PROBABILITIES)
    samples = simulate_discrete(VALUES, PROBABILITIES)
    empirical = empirical_proportions(samples, VALUES)
    cumulative = np.array([discrete_cdf(VALUES, PROBABILITIES, x) for x in VALUES])
    print("Chapter 16 — Random Variables")
    print("Uncertain business quantity: X = number of restaurant customers tomorrow.")
    print("Possible values and model PMF:")
    print("  " + "  ".join(f"{x}: {p:.2f}" for x, p in zip(VALUES, PROBABILITIES, strict=True)))
    print(f"PMF valid: nonnegative and sum={PROBABILITIES.sum():.2f}.")
    print(f"P(X=160)={exact_probability(VALUES, PROBABILITIES, 160):.2f}; P(X≥200)={threshold_probability(VALUES, PROBABILITIES, 200):.2f}; P(X<160)={threshold_probability(VALUES, PROBABILITIES, 160, comparison='<'):.2f}.")
    print(f"P(120≤X≤200)={probability_between(VALUES, PROBABILITIES, 120, 200):.2f}; P(X≠160)={1-exact_probability(VALUES, PROBABILITIES, 160):.2f}.")
    print("CDF at the support: " + ", ".join(f"F({x})={p:.2f}" for x, p in zip(VALUES, cumulative, strict=True)))
    print(f"High-demand indicator I=1 when X≥200, so P(I=1)={threshold_probability(VALUES, PROBABILITIES, 200):.2f}.")
    print(f"Transformation R=18X: X=160 maps to revenue R=${revenue_transform([160])[0]:,.0f}; excess-capacity W=max(X-180,0) maps support to {np.maximum(VALUES-180, 0).tolist()}.")
    print("Fixed-seed simulation (1,000 realizations): " + ", ".join(f"{x}: {p:.3f}" for x, p in zip(VALUES, empirical, strict=True)))
    paths = create_figures(output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures: outcome mapping, PMF, CDF, model-versus-sample, and continuous preview.")
    print("Interpretation: the PMF is the probability model; empirical proportions describe this simulated sample and need not match it exactly.")
    print("Continuous preview: Y = order preparation time; P(Y=10)=0, while interval probability is represented by area under a curve.")
    print("Next, Chapter 17 will introduce named distributions; expected value is deliberately deferred to Chapter 18.")
    return 0
