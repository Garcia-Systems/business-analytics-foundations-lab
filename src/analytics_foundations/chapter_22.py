"""Sampling distributions, standard error, and the CLT for Chapter 22."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.datasets import PROJECT_ROOT

REQUIRED_FIGURES = (
    "chapter-22-three-distributions.png",
    "chapter-22-sampling-mean.png",
    "chapter-22-sample-size.png",
    "chapter-22-empirical-theoretical-se.png",
    "chapter-22-clt.png",
    "chapter-22-bias-precision.png",
)


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def simulate_sample_means(population: np.ndarray, *, sample_size: int,
                          n_repetitions: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Return one mean per independent, with-replacement simulated sample."""
    values = np.asarray(population, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("population must be nonempty")
    _positive_integer(sample_size, "sample_size")
    _positive_integer(n_repetitions, "n_repetitions")
    draws = rng.choice(values, size=(n_repetitions, sample_size), replace=True)
    return draws.mean(axis=1)


def standard_error_mean(population_sd: float, sample_size: int) -> float:
    """Calculate sigma / sqrt(n) for independent observations."""
    _positive_integer(sample_size, "sample_size")
    if not np.isfinite(population_sd) or population_sd < 0:
        raise ValueError("population_sd must be finite and nonnegative")
    return float(population_sd / np.sqrt(sample_size))


def estimated_standard_error_mean(sample_sd: float, sample_size: int) -> float:
    """Estimate a mean's standard error with the observed sample SD."""
    return standard_error_mean(sample_sd, sample_size)


def standard_error_proportion(p: float, sample_size: int) -> float:
    """Calculate sqrt(p(1-p)/n) for independent Bernoulli observations."""
    _positive_integer(sample_size, "sample_size")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    return float(np.sqrt(p * (1 - p) / sample_size))


def simulate_sample_proportions(p: float, *, sample_size: int,
                                n_repetitions: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Simulate sampling distributions of a binary sample proportion."""
    standard_error_proportion(p, sample_size)
    _positive_integer(n_repetitions, "n_repetitions")
    return rng.binomial(sample_size, p, size=n_repetitions) / sample_size


def generate_skewed_population(*, size: int = 5_000, seed: int = 2222) -> np.ndarray:
    """Create a deterministic, visibly right-skewed finite population."""
    _positive_integer(size, "size")
    return np.random.default_rng(seed).lognormal(mean=2.3, sigma=1.0, size=size)


def sample_size_experiment(population: np.ndarray, *, sizes=(5, 20, 50, 200),
                           n_repetitions: int = 10_000, seed: int = 2201) -> pd.DataFrame:
    """Compare empirical sampling SD with sigma/sqrt(n)."""
    values = np.asarray(population, dtype=float)
    sigma = values.std(ddof=0)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        means = simulate_sample_means(values, sample_size=n,
                                      n_repetitions=n_repetitions, rng=rng)
        rows.append({"n": n, "empirical_se": means.std(ddof=0),
                     "theoretical_se": standard_error_mean(sigma, n),
                     "mean_of_means": means.mean()})
    return pd.DataFrame(rows)


def biased_sampling_experiment(population: np.ndarray, *, sample_size: int = 200,
                               n_repetitions: int = 5_000, bias: float = 5.0,
                               seed: int = 2202) -> tuple[np.ndarray, np.ndarray]:
    """Contrast unbiased estimates with precise estimates shifted from truth."""
    rng = np.random.default_rng(seed)
    unbiased = simulate_sample_means(population, sample_size=sample_size,
                                     n_repetitions=n_repetitions, rng=rng)
    return unbiased, unbiased + bias


def _skewness(values: np.ndarray) -> float:
    centered = values - values.mean()
    return float(np.mean(centered ** 3) / np.mean(centered ** 2) ** 1.5)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(population: np.ndarray, one_sample: np.ndarray,
                   sample_means: np.ndarray, output_dir: Path) -> list[Path]:
    """Generate the chapter's six visual explanations."""
    output_dir = Path(output_dir); paths = []; mu = population.mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title in zip(axes, [population, one_sample, sample_means],
                              ["Population: individual X", "One sample: observed X",
                               "Sampling distribution: sample mean"]):
        ax.hist(data, bins=25, color="#4c78a8"); ax.set_title(title); ax.set_xlabel("Minutes")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(sample_means, bins=35, color="#59a14f")
    ax.axvline(mu, color="black", ls="--", label="population mean")
    ax.axvline(sample_means.mean(), color="#e45756", label="mean of sample means")
    ax.set(title="Sampling distribution of the mean (n=40)", xlabel="Sample mean"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))
    rng = np.random.default_rng(2203); size_means = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, n in zip(axes.flat, (5, 20, 50, 200)):
        size_means[n] = simulate_sample_means(population, sample_size=n, n_repetitions=4000, rng=rng)
        ax.hist(size_means[n], bins=30); ax.axvline(mu, color="black", ls="--"); ax.set_title(f"n={n}")
    fig.supxlabel("Sample mean (same scale)")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))
    comparison = sample_size_experiment(population, n_repetitions=5000)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(comparison.n, comparison.empirical_se, "o-", label="empirical SD")
    ax.plot(comparison.n, comparison.theoretical_se, "s--", label="theoretical SE")
    ax.set(title="Standard error shrinks with 1/sqrt(n)", xlabel="n", ylabel="Minutes"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))
    skewed = generate_skewed_population(); rng = np.random.default_rng(2204)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].hist(skewed, bins=35); axes[0].set_title("Raw population")
    for ax, n in zip(axes[1:], (1, 5, 30, 100)):
        means = simulate_sample_means(skewed, sample_size=n, n_repetitions=5000, rng=rng)
        ax.hist(means, bins=35); ax.set_title(f"Means, n={n}")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))
    rng = np.random.default_rng(2205)
    wide = simulate_sample_means(population, sample_size=10, n_repetitions=3000, rng=rng)
    narrow, biased = biased_sampling_experiment(population, seed=2206)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.hist(wide, bins=35, alpha=.45, label="unbiased, small n")
    ax.hist(narrow, bins=35, alpha=.55, label="unbiased, large n"); ax.hist(biased, bins=35, alpha=.55, label="biased, large n")
    ax.axvline(mu, color="black", ls="--", label="truth"); ax.set(title="Precision is not accuracy", xlabel="Estimate"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 22 repeated-sampling experiment."""
    population = generate_population().wait_minutes.to_numpy(); mu = population.mean(); sigma = population.std(ddof=0)
    rng = np.random.default_rng(22); sample = rng.choice(population, 40, replace=True)
    means = simulate_sample_means(population, sample_size=40, n_repetitions=10_000, rng=rng)
    comparison = sample_size_experiment(population)
    p = np.mean(population > 20); proportions = {n: simulate_sample_proportions(p, sample_size=n, n_repetitions=5000, rng=rng).std(ddof=0) for n in (25, 100, 400)}
    skewed = generate_skewed_population(); clt = {}
    for n in (1, 5, 30, 100):
        clt[n] = _skewness(simulate_sample_means(skewed, sample_size=n, n_repetitions=5000, rng=rng))
    paths = create_figures(population, sample, means, output_dir or PROJECT_ROOT / "figures")
    print("Chapter 22 — Sampling Distributions")
    print(f"Population: N={len(population):,}; μ={mu:.2f} min; σ={sigma:.2f} min")
    print("In real inference, population parameters are usually unknown; here they are known so we can test the theory.")
    print(f"One sample: n=40; x̄={sample.mean():.2f} min")
    print(f"Repeated sampling (10,000 means): center={means.mean():.2f}; empirical SE={means.std(ddof=0):.3f}; theoretical SE={standard_error_mean(sigma, 40):.3f}")
    print("Sample-size comparison (n: empirical / theoretical SE):")
    print("  " + "; ".join(f"{int(r.n)}: {r.empirical_se:.3f} / {r.theoretical_se:.3f}" for r in comparison.itertuples()))
    print("Proportion wait>20 empirical SE (n=25/100/400): " + ", ".join(f"{n}: {se:.3f}" for n, se in proportions.items()))
    print("CLT with right-skewed population (skewness of sample means): " + ", ".join(f"n={n}: {value:.2f}" for n, value in clt.items()))
    print("The CLT concerns sample means, not raw data; n=30 is not a universal guarantee.")
    print("Assumptions: appropriate sampling, reasonable independence, a stable process, and finite variance; clustering can make the effective information smaller than the row count.")
    print("Key lesson: x̄ is random before sampling; standard error is its sampling SD, falls as 1/sqrt(n), and is not raw-data SD. Large n reduces variation, not bias.")
    print(f"Generated {len(paths)} figures. Chapter 23 will ask how one estimate and its estimated SE can describe plausible population values; no interval is built here.")
    return 0
