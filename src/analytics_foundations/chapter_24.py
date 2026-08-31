"""One-sample hypothesis testing for Chapter 24."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, ttest_1samp

from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.chapter_23 import mean_confidence_interval
from analytics_foundations.datasets import PROJECT_ROOT

ALTERNATIVES = frozenset({"greater", "less", "two-sided"})
REQUIRED_FIGURES = (
    "chapter-24-null-distribution.png",
    "chapter-24-one-vs-two-sided.png",
    "chapter-24-null-rejections.png",
    "chapter-24-power.png",
    "chapter-24-business-significance.png",
    "chapter-24-ci-null.png",
)


@dataclass(frozen=True)
class HypothesisTestResult:
    """Inspectable pieces of a manually calculated one-sample t test."""

    sample_size: int
    sample_mean: float
    sample_sd: float
    standard_error: float
    null_mean: float
    effect: float
    t_statistic: float
    degrees_of_freedom: int
    p_value: float
    alternative: str


def one_sample_t_test(values: np.ndarray, *, null_mean: float,
                      alternative: str = "two-sided") -> HypothesisTestResult:
    """Calculate a one-sample t statistic and the requested tail probability."""
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {sorted(ALTERNATIVES)}")
    if not np.isfinite(null_mean):
        raise ValueError("null_mean must be finite")
    sample = np.asarray(values, dtype=float).reshape(-1)
    if sample.size < 2:
        raise ValueError("at least two observations are required")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must all be finite")
    n = sample.size
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1))
    if sd == 0:
        raise ValueError("sample standard deviation must be positive")
    se = sd / np.sqrt(n)
    effect = mean - null_mean
    statistic = effect / se
    df = n - 1
    if alternative == "greater":
        p_value = t.sf(statistic, df=df)
    elif alternative == "less":
        p_value = t.cdf(statistic, df=df)
    else:
        p_value = 2 * t.sf(abs(statistic), df=df)
    return HypothesisTestResult(n, mean, sd, se, float(null_mean), effect,
                                float(statistic), df, float(p_value), alternative)


def reject_null(p_value: float, alpha: float = .05) -> bool:
    """Apply a prespecified threshold without replacing substantive interpretation."""
    if not np.isfinite(p_value) or not 0 <= p_value <= 1:
        raise ValueError("p_value must be between 0 and 1")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return bool(p_value <= alpha)


def simulate_rejection_rate(*, true_mean: float, null_mean: float = 15,
                            population_sd: float = 8, sample_size: int = 64,
                            repetitions: int = 5000, alpha: float = .05,
                            seed: int = 2401) -> float:
    """Estimate right-tailed rejection probability with reproducible Normal draws."""
    if sample_size < 2 or repetitions < 1:
        raise ValueError("sample_size >= 2 and repetitions >= 1 are required")
    if not np.isfinite(population_sd) or population_sd <= 0:
        raise ValueError("population_sd must be positive and finite")
    reject_null(.5, alpha)  # validate alpha consistently
    draws = np.random.default_rng(seed).normal(true_mean, population_sd,
                                                size=(repetitions, sample_size))
    means = draws.mean(axis=1)
    ses = draws.std(axis=1, ddof=1) / np.sqrt(sample_size)
    statistics = (means - null_mean) / ses
    p_values = t.sf(statistics, df=sample_size - 1)
    return float(np.mean(p_values <= alpha))


def sample_size_signal(*, effect: float = 1, sample_sd: float = 8,
                       sizes: tuple[int, ...] = (16, 64, 400)) -> list[tuple[int, float, float]]:
    """Hold signal and variability fixed to expose the sample-size mechanism."""
    if sample_sd <= 0 or any(n < 2 for n in sizes):
        raise ValueError("sample_sd must be positive and sizes must be at least two")
    return [(n, sample_sd / np.sqrt(n), effect / (sample_sd / np.sqrt(n))) for n in sizes]


def outlier_comparison(values: np.ndarray, *, null_mean: float = 15,
                       extreme_wait: float = 80) -> tuple[HypothesisTestResult, HypothesisTestResult]:
    """Compare the original test with a test after appending one extreme wait."""
    sample = np.asarray(values, dtype=float).reshape(-1)
    return (one_sample_t_test(sample, null_mean=null_mean, alternative="greater"),
            one_sample_t_test(np.append(sample, extreme_wait), null_mean=null_mean,
                              alternative="greater"))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(result: HypothesisTestResult, values: np.ndarray,
                   output_dir: Path) -> list[Path]:
    """Create six views of tails, errors, power, importance, and the CI link."""
    output_dir = Path(output_dir); paths: list[Path] = []
    x = np.linspace(-4.5, 4.5, 800); density = t.pdf(x, result.degrees_of_freedom)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, density); ax.fill_between(x, density, where=x >= result.t_statistic, alpha=.4, label=f"p = {result.p_value:.3f}")
    ax.axvline(0, color="black", linestyle=":", label="null center"); ax.axvline(result.t_statistic, color="C3", label="observed t")
    ax.set(title="Evidence more extreme under the null", xlabel="t statistic", ylabel="Density"); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, two in zip(axes, (False, True)):
        ax.plot(x, density); mask = (x >= result.t_statistic) | (two & (x <= -abs(result.t_statistic)))
        ax.fill_between(x, density, where=mask, alpha=.4); ax.axvline(result.t_statistic, color="C3")
        ax.set(title="Two-sided: both tails" if two else "Right-sided: upper tail", xlabel="t")
    axes[0].set_ylabel("Density"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    seeds = range(30); rates = [simulate_rejection_rate(true_mean=15, repetitions=1000, seed=s) for s in seeds]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.scatter(list(seeds), rates); ax.axhline(.05, color="C3", label="α = 0.05")
    ax.set(title="Type I rejection rates fluctuate around α", xlabel="Repeated simulation batch", ylabel="Rejection rate", ylim=(0, .1)); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    sizes = (16, 64, 256); powers = [simulate_rejection_rate(true_mean=17, sample_size=n, seed=2424+n) for n in sizes]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(sizes, powers, marker="o"); ax.set(title="Simulated power rises with sample size (true μ=17)", xlabel="Sample size", ylabel="Rejection probability", ylim=(0, 1)); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))

    cases = ("Large effect\nsmall n", "Tiny effect\nhuge n"); effects = (2.0, .2)
    fig, ax = plt.subplots(figsize=(7, 4)); bars=ax.bar(cases, effects, color=("C0", "C2")); ax.bar_label(bars, labels=("2.0 min", "0.2 min = 12 sec")); ax.set(title="Significance does not determine business importance", ylabel="Raw effect above benchmark (minutes)"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))

    interval = mean_confidence_interval(values)
    fig, ax = plt.subplots(figsize=(8, 2.8)); ax.errorbar(interval.estimate, 0, xerr=interval.margin_of_error, fmt="o", capsize=8, label="95% two-sided CI"); ax.axvline(result.null_mean, color="C3", linestyle="--", label="null μ₀")
    ax.set(title="A matching two-sided test and 95% interval agree", xlabel="Mean wait (minutes)", yticks=[]); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 24 restaurant benchmark experiment."""
    population = generate_population().wait_minutes.to_numpy()
    sample = np.random.default_rng(23).choice(population, 64, replace=True)
    result = one_sample_t_test(sample, null_mean=15, alternative="greater")
    scipy_result = ttest_1samp(sample, popmean=15, alternative="greater")
    two_sided = one_sample_t_test(sample, null_mean=15, alternative="two-sided")
    interval = mean_confidence_interval(sample)
    type_i = simulate_rejection_rate(true_mean=15)
    powers = {n: simulate_rejection_rate(true_mean=17, sample_size=n, seed=2400+n) for n in (16, 64, 256)}
    paths = create_figures(result, sample, output_dir or PROJECT_ROOT / "figures")
    decision = "reject H0" if reject_null(result.p_value) else "fail to reject H0"
    consistency = (two_sided.p_value <= .05) == (not interval.lower <= 15 <= interval.upper)
    print("Chapter 24 — Hypothesis Testing")
    print("Business question: Is mean Friday dinner wait time greater than 15 minutes?")
    print("H0: mu = 15; HA: mu > 15 (chosen before seeing the result)")
    print(f"Sample: n={result.sample_size}; mean={result.sample_mean:.2f}; sd={result.sample_sd:.2f}; SE={result.standard_error:.3f} minutes")
    print(f"Test: t({result.degrees_of_freedom})={result.t_statistic:.3f}; one-sided p={result.p_value:.4f}; alpha=0.05; decision={decision}")
    print(f"Manual/SciPy check: statistic difference={abs(result.t_statistic-scipy_result.statistic):.2g}; p-value difference={abs(result.p_value-scipy_result.pvalue):.2g}")
    print(f"Effect: sample mean - benchmark = {result.effect:.2f} minutes")
    print(f"95% two-sided CI: [{interval.lower:.2f}, {interval.upper:.2f}]; matching two-sided p={two_sided.p_value:.4f}; CI/test conclusions consistent={consistency}")
    print(f"Tail choice: right-sided p={result.p_value:.4f}; two-sided p={two_sided.p_value:.4f}. A one-sided test answers a different prespecified question, not a request for a smaller p-value.")
    print("Sample-size mechanism (n: SE, t for a fixed 1-minute effect): " + "; ".join(f"{n}: {se:.2f}, {stat:.2f}" for n,se,stat in sample_size_signal()))
    print(f"Simulation: true-null rejection rate={type_i:.1%}; simulated power for true mu=17: " + ", ".join(f"n={n}: {power:.1%}" for n,power in powers.items()))
    print("Business interpretation: the p-value measures incompatibility with the null model, not P(H0 | data). Pair evidence with the effect in minutes; statistical significance need not mean operational importance.")
    print("Errors: Type I spends on changes when mu=15; Type II misses a meaningful exceedance. Their business costs should inform alpha; 0.05 is not universal.")
    print("Limitations: define the target population; require representative, reasonably independent, reliable observations and a stable process. Inspect skew/outliers. More data cannot repair selection bias, measurement error, data snooping, or repeated testing.")
    print(f"Generated {len(paths)} figures. Next question: are observed differences among groups larger than ordinary sampling variation? That belongs to Chapter 25; no group test is performed here.")
    return 0
