"""Expected value, variability, and restaurant payoff models for Chapter 18."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from analytics_foundations.chapter_16 import simulate_discrete, validate_pmf
from analytics_foundations.datasets import PROJECT_ROOT

PROFIT_VALUES = np.array([-1000., 500., 2000., 4000.])
PROFIT_PROBABILITIES = np.array([.15, .35, .30, .20])
REQUIRED_FIGURES = (
    "chapter-18-weighted-balance.png",
    "chapter-18-same-mean-different-spread.png",
    "chapter-18-profit-deviations.png",
    "chapter-18-simulation-mean.png",
    "chapter-18-simulation-variance.png",
)


def _validated(values: ArrayLike, probabilities: ArrayLike) -> tuple[NDArray, NDArray]:
    values_array = np.asarray(values, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    validate_pmf(values_array, probabilities_array)
    return values_array, probabilities_array


def expected_value(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the probability-weighted average of a discrete PMF."""
    values_array, probabilities_array = _validated(values, probabilities)
    return float(np.sum(values_array * probabilities_array))


def discrete_variance(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the weighted average squared deviation from the model mean."""
    values_array, probabilities_array = _validated(values, probabilities)
    mu = expected_value(values_array, probabilities_array)
    return float(np.sum(((values_array - mu) ** 2) * probabilities_array))


def variance_shortcut(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return E[X^2] - E[X]^2 for a validated discrete PMF."""
    values_array, probabilities_array = _validated(values, probabilities)
    mu = expected_value(values_array, probabilities_array)
    return float(np.sum(values_array**2 * probabilities_array) - mu**2)


def discrete_standard_deviation(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return discrete model spread in the random variable's original units."""
    return float(np.sqrt(discrete_variance(values, probabilities)))


def contribution_table(values: ArrayLike, probabilities: ArrayLike) -> pd.DataFrame:
    """Build an auditable expectation and variance calculation table."""
    values_array, probabilities_array = _validated(values, probabilities)
    mu = expected_value(values_array, probabilities_array)
    deviations = values_array - mu
    return pd.DataFrame({
        "outcome": values_array,
        "probability": probabilities_array,
        "weighted_value": values_array * probabilities_array,
        "deviation": deviations,
        "squared_deviation": deviations**2,
        "weighted_squared_deviation": deviations**2 * probabilities_array,
    })


def simulate_profit(size: int = 10_000, *, seed: int = 18) -> NDArray:
    """Draw reproducible special-event profits from the primary PMF."""
    return simulate_discrete(PROFIT_VALUES, PROFIT_PROBABILITIES, size, seed=seed)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(output_dir: Path, *, seed: int = 18) -> list[Path]:
    """Generate balance, risk, deviation, and simulation figures."""
    output_dir = Path(output_dir); paths: list[Path] = []
    mu = expected_value(PROFIT_VALUES, PROFIT_PROBABILITIES)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.stem(PROFIT_VALUES, PROFIT_PROBABILITIES)
    ax.axvline(mu, color="#e45756", ls="--", label=f"E[X] = ${mu:,.0f}")
    ax.set(xlabel="Special-event profit ($)", ylabel="Probability", title="Expected profit is the PMF's weighted balance point"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    stable, risky = np.array([900, 1100]), np.array([-1000, 3000]); probs = np.array([.5, .5])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, values, name, color in zip(axes, [stable, risky], ["Stable", "Risky"], ["#4c78a8", "#f28e2b"], strict=True):
        ax.bar(values, probs, width=180 if name == "Stable" else 500, color=color)
        ax.axvline(1000, color="black", ls="--"); ax.set_title(f"{name}: mean $1,000\nSD ${discrete_standard_deviation(values, probs):,.0f}")
        ax.set(xlabel="Profit ($)", ylim=(0, .65))
    axes[0].set_ylabel("Probability"); fig.suptitle("Same expected reward, very different spread")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    fig, ax = plt.subplots(figsize=(9, 3)); ax.scatter(PROFIT_VALUES, np.zeros(4), s=80); ax.axvline(mu, color="#e45756", ls="--")
    for x in PROFIT_VALUES: ax.annotate(f"{x-mu:+,.0f}", ((x+mu)/2, .04), ha="center")
    ax.set(yticks=[], xlabel="Profit ($)", title="Signed distances from expected profit (squaring prevents cancellation)")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    samples = simulate_profit(10_000, seed=seed); n = np.arange(1, len(samples)+1)
    cumulative_mean = np.cumsum(samples) / n
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(n, cumulative_mean, lw=1); ax.axhline(mu, color="#e45756", ls="--", label="theoretical E[X]")
    ax.set(xscale="log", xlabel="Repeated events", ylabel="Cumulative mean profit ($)", title="A simulated average moves around the model expectation"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))

    checkpoints = np.unique(np.geomspace(10, len(samples), 150).astype(int)); empirical = [samples[:i].var(ddof=0) for i in checkpoints]
    variance = discrete_variance(PROFIT_VALUES, PROFIT_PROBABILITIES)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(checkpoints, empirical); ax.axhline(variance, color="#e45756", ls="--", label="theoretical Var(X)")
    ax.set(xscale="log", xlabel="Repeated events", ylabel="Empirical variance ($²)", title="Empirical spread compared with model spread"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4])); return paths


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 18 restaurant reward-and-risk experiment."""
    table = contribution_table(PROFIT_VALUES, PROFIT_PROBABILITIES)
    mu = expected_value(PROFIT_VALUES, PROFIT_PROBABILITIES); variance = discrete_variance(PROFIT_VALUES, PROFIT_PROBABILITIES); sd = np.sqrt(variance)
    samples = simulate_profit(); sizes = (10, 100, 1000, 10000)
    print("Chapter 18 — Expected Value & Variability")
    print("Special-event PMF (validated); each row makes the weighted calculations auditable:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"E[X]=(-1000)(.15)+(500)(.35)+(2000)(.30)+(4000)(.20)=${mu:,.2f}.")
    print(f"Var(X)=Σ(x-μ)²P(X=x)={variance:,.2f} dollars²; SD(X)=sqrt(Var(X))=${sd:,.2f}.")
    print(f"Shortcut check: E[X²]-E[X]²={variance_shortcut(PROFIT_VALUES, PROFIT_PROBABILITIES):,.2f}.")
    transformed = 3*PROFIT_VALUES+5
    print(f"Y=3X+5: E[Y]={expected_value(transformed, PROFIT_PROBABILITIES):,.2f}=3E[X]+5; Var(Y)={discrete_variance(transformed, PROFIT_PROBABILITIES):,.2f}=9Var(X).")
    print("Same mean, different risk: stable {900,1100} and risky {-1000,3000} both average $1,000; their SDs are $100 and $2,000.")
    p, n = .20, 20
    print(f"Bernoulli({p}): E[X]=p={p:.2f}, Var(X)=p(1-p)={p*(1-p):.2f}; Binomial({n},{p}): E[X]=np={n*p:.1f}, Var(X)=np(1-p)={n*p*(1-p):.1f}.")
    print("Normal N(μ,σ²) has E[X]=μ, Var(X)=σ², and SD(X)=σ; Uniform(a,b) has mean (a+b)/2 and variance (b-a)²/12.")
    print("Fixed-seed empirical means (not monotonic): " + ", ".join(f"n={size}: {samples[:size].mean():,.1f}" for size in sizes))
    print(f"At n=10,000: empirical mean={samples.mean():,.1f} vs theoretical μ={mu:,.1f}; empirical variance={samples.var(ddof=0):,.0f} vs theoretical σ²={variance:,.0f}.")
    paths = create_figures(output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures. Expected value summarizes reward; SD summarizes one dimension of risk—neither alone decides.")
    print("Model E[X], Var(X) are theoretical parameters; sample mean and ddof=0 empirical variance summarize realizations. Dependence changes variance-of-sums reasoning; Chapter 19 takes that next step.")
    return 0
