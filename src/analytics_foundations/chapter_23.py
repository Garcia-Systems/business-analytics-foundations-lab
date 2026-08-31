"""Point estimates and confidence intervals for Chapter 23."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.proportion import proportion_confint

from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.datasets import PROJECT_ROOT

REQUIRED_FIGURES = (
    "chapter-23-interval-anatomy.png", "chapter-23-coverage.png",
    "chapter-23-confidence-width.png", "chapter-23-sample-size-width.png",
    "chapter-23-t-versus-normal.png", "chapter-23-individual-versus-mean.png",
)


@dataclass(frozen=True)
class MeanInterval:
    """Inspectable components of a one-sample t confidence interval."""

    estimate: float
    sample_sd: float
    standard_error: float
    critical_value: float
    margin_of_error: float
    lower: float
    upper: float
    confidence: float
    degrees_of_freedom: int


def _validate_confidence(confidence: float) -> None:
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def mean_confidence_interval(values: np.ndarray, *, confidence: float = .95) -> MeanInterval:
    """Construct the default t interval for a mean when sigma is unknown."""
    _validate_confidence(confidence)
    sample = np.asarray(values, dtype=float).reshape(-1)
    if sample.size < 2:
        raise ValueError("at least two observations are required")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must all be finite")
    n = sample.size
    estimate = float(sample.mean())
    sample_sd = float(sample.std(ddof=1))
    standard_error = sample_sd / np.sqrt(n)
    critical = float(t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    margin = critical * standard_error
    return MeanInterval(estimate, sample_sd, standard_error, critical, margin,
                        estimate - margin, estimate + margin, confidence, n - 1)


def simulate_mean_interval_coverage(population: np.ndarray, *, sample_size: int,
                                    confidence: float, repetitions: int,
                                    rng: np.random.Generator) -> pd.DataFrame:
    """Simulate t intervals and record whether each covers the fixed truth."""
    _validate_confidence(confidence)
    values = np.asarray(population, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("population must be nonempty")
    if sample_size < 2 or repetitions < 1:
        raise ValueError("sample_size >= 2 and repetitions >= 1 are required")
    draws = rng.choice(values, size=(repetitions, sample_size), replace=True)
    means = draws.mean(axis=1)
    ses = draws.std(axis=1, ddof=1) / np.sqrt(sample_size)
    critical = t.ppf(1 - (1 - confidence) / 2, df=sample_size - 1)
    margins = critical * ses
    lower, upper = means - margins, means + margins
    truth = values.mean()
    return pd.DataFrame({"sample_mean": means, "standard_error": ses,
                         "lower": lower, "upper": upper,
                         "covers_true_mean": (lower <= truth) & (truth <= upper)})


def required_sample_size(*, population_sd: float, margin_of_error: float,
                         confidence: float = .95) -> int:
    """Normal-approximation planning size, always rounded upward."""
    _validate_confidence(confidence)
    if not np.isfinite(population_sd) or population_sd <= 0:
        raise ValueError("population_sd must be positive and finite")
    if not np.isfinite(margin_of_error) or margin_of_error <= 0:
        raise ValueError("margin_of_error must be positive and finite")
    critical = norm.ppf(1 - (1 - confidence) / 2)
    return int(np.ceil((critical * population_sd / margin_of_error) ** 2))


def wilson_proportion_interval(successes: int, n: int, *, confidence: float = .95) -> tuple[float, float]:
    """Return a better-behaved Wilson interval for a binomial proportion."""
    _validate_confidence(confidence)
    if not isinstance(successes, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("successes and n must be integers")
    if n < 1 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n and n >= 1")
    low, high = proportion_confint(successes, n, alpha=1-confidence, method="wilson")
    return float(low), float(high)


def biased_interval_experiment(population: np.ndarray, *, sample_size: int = 1000,
                               bias: float = 5, seed: int = 2304) -> tuple[MeanInterval, float]:
    """Create a narrow interval shifted away from the actual population mean."""
    values = np.asarray(population, dtype=float)
    sample = np.random.default_rng(seed).choice(values, sample_size, replace=True) + bias
    return mean_confidence_interval(sample), float(values.mean())


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(population: np.ndarray, sample: np.ndarray, coverage: pd.DataFrame,
                   output_dir: Path) -> list[Path]:
    """Generate six conceptual confidence-interval figures."""
    output_dir = Path(output_dir); paths = []; interval = mean_confidence_interval(sample)
    fig, ax = plt.subplots(figsize=(8, 2.8)); ax.errorbar(interval.estimate, 0, xerr=interval.margin_of_error, fmt="o", capsize=8)
    ax.annotate("lower", (interval.lower, .05)); ax.annotate("estimate", (interval.estimate, .05), ha="center"); ax.annotate("upper", (interval.upper, .05), ha="right")
    ax.set(title="estimate ± critical value × standard error", xlabel="Wait time (minutes)", yticks=[])
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))
    shown = coverage.head(50); truth = np.asarray(population).mean(); fig, ax = plt.subplots(figsize=(9, 7))
    for i, row in enumerate(shown.itertuples()):
        style = "-" if row.covers_true_mean else "--"
        marker = "o" if row.covers_true_mean else "x"
        ax.plot([row.lower, row.upper], [i, i], linestyle=style, marker=marker, markevery=[0])
    ax.axvline(truth, color="black", linestyle=":", label="fixed population mean")
    ax.set(title="Intervals move; the parameter does not", xlabel="Mean wait (minutes)", ylabel="Simulated sample"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))
    levels = (.90, .95, .99); intervals = [mean_confidence_interval(sample, confidence=c) for c in levels]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, result in enumerate(intervals): ax.errorbar(result.estimate, i, xerr=result.margin_of_error, fmt="o", capsize=6)
    ax.set(yticks=range(3), yticklabels=[f"{c:.0%}" for c in levels], title="Higher confidence means a wider interval", xlabel="Mean wait (minutes)")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))
    sizes = np.arange(4, 401); margins = t.ppf(.975, sizes-1) * np.asarray(population).std(ddof=0) / np.sqrt(sizes)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(sizes, margins); ax.set(title="95% margin shrinks roughly as 1/√n", xlabel="Sample size n", ylabel="Margin (minutes)")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))
    x = np.linspace(-4, 4, 500); fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, norm.pdf(x), label="Normal"); ax.plot(x, t.pdf(x, 4), "--", label="t, df=4"); ax.plot(x, t.pdf(x, 29), ":", label="t, df=29"); ax.set(title="Estimating σ adds tail uncertainty", xlabel="Standardized value", ylabel="Density"); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))
    fig, ax = plt.subplots(figsize=(9, 3)); y = np.zeros(sample.size); ax.scatter(sample, y, alpha=.45, label="individual waits"); ax.errorbar(interval.estimate, .25, xerr=interval.margin_of_error, fmt="o", capsize=7, label="95% CI for mean"); ax.set(title="Individual variation is not uncertainty about the mean", xlabel="Minutes", yticks=[]); ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 23 estimation experiment."""
    population = generate_population().wait_minutes.to_numpy(); rng = np.random.default_rng(23)
    sample = rng.choice(population, 64, replace=True); result = mean_confidence_interval(sample)
    z_margin = norm.ppf(.975) * result.standard_error
    comparisons = {c: mean_confidence_interval(sample, confidence=c) for c in (.90, .95, .99)}
    coverage = simulate_mean_interval_coverage(population, sample_size=64, confidence=.95,
                                               repetitions=5000, rng=np.random.default_rng(2301))
    biased, truth = biased_interval_experiment(population)
    successes = int(np.sum(sample > 20)); proportion = wilson_proportion_interval(successes, len(sample))
    paths = create_figures(population, sample, coverage, output_dir or PROJECT_ROOT / "figures")
    print("Chapter 23 — Estimation & Confidence")
    print(f"Sample: n=64; mean={result.estimate:.2f} min; sd={result.sample_sd:.2f} min; estimated SE={result.standard_error:.3f} min")
    print(f"95% normal intuition: estimate ± 1.960×SE = [{result.estimate-z_margin:.2f}, {result.estimate+z_margin:.2f}]")
    print(f"95% t interval (df={result.degrees_of_freedom}): [{result.lower:.2f}, {result.upper:.2f}]; margin={result.margin_of_error:.2f} min")
    print("Confidence comparison (level: margin / width): " + "; ".join(f"{c:.0%}: {r.margin_of_error:.2f} / {2*r.margin_of_error:.2f}" for c, r in comparisons.items()))
    print(f"Coverage experiment: nominal=95%; empirical={coverage.covers_true_mean.mean():.1%}; misses={(~coverage.covers_true_mean).sum()} of 5,000")
    print("The intervals change across samples; the population mean remains fixed. Confidence is the procedure's long-run coverage under its assumptions.")
    print(f"Biased large sample: narrow interval [{biased.lower:.2f}, {biased.upper:.2f}] misses true μ={truth:.2f}; precision does not repair bias.")
    print(f"Wait >20 minutes: {successes}/64; 95% Wilson proportion interval=[{proportion[0]:.3f}, {proportion[1]:.3f}]")
    print(f"Sample-size planning: σ≈8, target margin=1 minute, 95% normal planning requires n={required_sample_size(population_sd=8, margin_of_error=1)}.")
    position = "includes values above" if result.upper > 15 else "lies entirely below"
    print(f"Business interpretation: the 95% interval {position} the 15-minute service target; report parameter uncertainty, not a probability about fixed μ.")
    print("Assumptions: appropriate target-population sample, reasonable independence, reliable measurement, and no severe small-sample skew/outliers. n≥30 is not a guarantee; intervals do not fix selection bias or bad data.")
    print(f"Generated {len(paths)} figures. Next: Chapter 24 asks how much evidence the sample provides against μ=15; no p-value or hypothesis test is performed here.")
    return 0
