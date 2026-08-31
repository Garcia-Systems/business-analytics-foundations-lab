"""Named probability distributions and business model checks for Chapter 17."""

from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, norm, uniform

from analytics_foundations.datasets import PROJECT_ROOT

REQUIRED_FIGURES = (
    "chapter-17-bernoulli-pmf.png", "chapter-17-binomial-pmf.png",
    "chapter-17-uniform-density.png", "chapter-17-normal-density.png",
    "chapter-17-normal-parameters.png", "chapter-17-binomial-simulation.png",
)


def bernoulli_probabilities(p: float) -> NDArray[np.float64]:
    """Return probabilities for outcomes 0 and 1."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    return np.array([1 - p, p])


def binomial_probability(k: int, n: int, p: float) -> float:
    """Calculate the Binomial PMF transparently from its three factors."""
    if not 0 <= k <= n or not 0 <= p <= 1:
        return 0.0
    return float(comb(n, k) * p**k * (1 - p) ** (n - k))


def binomial_tail(at_least: int, n: int, p: float) -> float:
    """Return P(X >= at_least) using the survival function."""
    return float(binom.sf(at_least - 1, n, p))


def uniform_model(a: float, b: float):
    """Return Uniform(a,b), using SciPy's scale=b-a convention."""
    if b <= a:
        raise ValueError("b must exceed a")
    return uniform(loc=a, scale=b - a)


def interval_probability(dist, lower: float, upper: float) -> float:
    """Return continuous interval probability as a CDF difference."""
    if upper < lower:
        raise ValueError("upper must not be below lower")
    return float(dist.cdf(upper) - dist.cdf(lower))


def z_score(x: float, mu: float, sigma: float) -> float:
    """Standardize a value relative to normal-model parameters."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (x - mu) / sigma


def simulate_distributions(size: int = 10_000, *, seed: int = 17) -> dict[str, NDArray]:
    """Draw reproducible realizations from all four chapter models."""
    if size <= 0:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    return {
        "bernoulli": rng.binomial(1, .20, size),
        "binomial": rng.binomial(20, .20, size),
        "uniform": rng.uniform(0, 10, size),
        "normal": rng.normal(12, 2, size),
    }


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(output_dir: Path, *, seed: int = 17) -> list[Path]:
    """Generate the six focused distribution figures required by the chapter."""
    output_dir = Path(output_dir); paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([0, 1], [.8, .2], color="#4c78a8")
    ax.set(xticks=[0, 1], xlabel="Offer redemption X", ylabel="P(X=x)", title="Bernoulli(0.20): one binary trial", ylim=(0, 1))
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    k = np.arange(21); pmf = binom.pmf(k, 20, .2)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(k, pmf, color="#4c78a8")
    ax.set(xticks=np.arange(0, 21, 2), xlabel="Redemptions k", ylabel="P(X=k)", title="Binomial(20, 0.20) PMF")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    x = np.linspace(-1, 11, 500); ud = uniform_model(0, 10); density = ud.pdf(x)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, density, lw=2); mask = (x >= 2) & (x <= 5)
    ax.fill_between(x[mask], density[mask], alpha=.6, color="#72b7b2", label="area = 0.30")
    ax.set(xlabel="Shuttle wait (minutes)", ylabel="Density (per minute)", title="Uniform(0, 10): probability is area"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    x = np.linspace(4, 20, 500); nd = norm(loc=12, scale=2); density = nd.pdf(x); mask = (x >= 10) & (x <= 14)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, density, lw=2); ax.fill_between(x[mask], density[mask], alpha=.6, color="#f28e2b")
    ax.set(xlabel="Preparation time (minutes)", ylabel="Density", title="N(12, 2²): area for 10 ≤ T ≤ 14")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))

    fig, ax = plt.subplots(figsize=(8, 4))
    for mu, sigma in [(10, 2), (12, 2), (12, 3)]: ax.plot(x, norm(mu, sigma).pdf(x), label=f"μ={mu}, σ={sigma}")
    ax.set(xlabel="Value", ylabel="Density", title="Normal parameters shift and spread the model"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))

    sample = simulate_distributions(50_000, seed=seed)["binomial"]; empirical = np.bincount(sample, minlength=21) / len(sample)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(k-.2, pmf, width=.4, label="Theoretical PMF"); ax.bar(k+.2, empirical, width=.4, label="Simulation", color="#f28e2b")
    ax.set(xlabel="Redemptions", ylabel="Probability / proportion", title="Model versus fixed-seed simulation"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5])); return paths


def run(output_dir: Path | None = None) -> int:
    """Run a concise tour from named models through assumptions and checks."""
    p = .2; exact = binomial_probability(2, 5, p); tail = binomial_tail(2, 5, p)
    ud = uniform_model(0, 10); nd = norm(loc=12, scale=2); samples = simulate_distributions()
    print("Chapter 17 — Distributions")
    print("X ~ D means random variable X follows distribution D; parameters describe the model, while observations are realized values.")
    print(f"Bernoulli(0.20), one offer: P(X=1)={p:.2f}, P(X=0)={1-p:.2f}; simulated redemption proportion={samples['bernoulli'].mean():.3f}.")
    print(f"Binomial(5, 0.20): P(X=2)=C(5,2)(.20)^2(.80)^3={exact:.4f}; P(X≥2)=1-P(X≤1)={tail:.4f}.")
    print("Binomial assumptions: fixed n; binary outcomes; common p; independence. Customer influence or segment-specific p breaks the model.")
    print(f"Uniform(0,10) shuttle wait: density=.10 per minute, P(2≤Y≤5)=3×.10={interval_probability(ud, 2, 5):.2f}; SciPy uses loc=0, scale=10.")
    print("Density is not point probability: P(Y=5)=0; continuous probability is area, equivalently a CDF difference.")
    print(f"Normal preparation model N(12,2²): P(T≤15)={nd.cdf(15):.4f}; P(10≤T≤14)={interval_probability(nd, 10, 14):.4f}; P(T>15)={nd.sf(15):.4f}.")
    print(f"Standardization: z=(15-12)/2={z_score(15, 12, 2):.1f}; 15 minutes is 1.5 standard deviations above the model center.")
    print(f"Simulation draws model realizations: Binomial sample P(X=4)≈{(samples['binomial']==4).mean():.3f}; Normal sample center≈{samples['normal'].mean():.2f}.")
    print("Uniform assumes equal density on its interval. Normal assumes plausible symmetric bell-shaped variation and assigns tiny probability to impossible negative times.")
    print("Mismatch check: right-skewed nonnegative transaction amounts should not automatically use Normal; simulation demonstrates assumptions, not realism.")
    paths = create_figures(output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures: Bernoulli PMF, Binomial PMF, Uniform area, Normal area, parameter comparison, and theory-versus-simulation.")
    print("Interpretation: named distributions are reusable uncertainty models only when their parameter claims and process assumptions are defensible.")
    return 0
