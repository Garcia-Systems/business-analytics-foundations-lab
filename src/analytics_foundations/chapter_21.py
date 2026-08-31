"""Samples, populations, and representativeness for Chapter 21."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT

POPULATION_SIZE = 5_000
POPULATION_SEED = 2121
REQUIRED_FIELDS = (
    "observation_id", "date", "location_id", "wait_minutes", "party_size",
    "reservation", "arrival_hour", "service_period",
)
REQUIRED_FIGURES = (
    "chapter-21-population-vs-sample.png",
    "chapter-21-several-samples.png",
    "chapter-21-sample-size.png",
    "chapter-21-size-vs-bias.png",
    "chapter-21-composition.png",
)


def generate_population(*, size: int = POPULATION_SIZE, seed: int = POPULATION_SEED) -> pd.DataFrame:
    """Create the deterministic, clean Friday-dinner teaching population."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    rng = np.random.default_rng(seed)
    locations = np.array(["Downtown", "Riverside", "West End", "Midtown", "Harbor"])
    location = rng.choice(locations, size=size, p=[.25, .20, .22, .18, .15])
    party_size = rng.choice(np.arange(1, 9), size=size,
                            p=[.08, .28, .22, .20, .10, .07, .03, .02])
    arrival_hour = rng.choice([17, 18, 19, 20, 21], size=size, p=[.16, .25, .29, .20, .10])
    reservation = rng.random(size) < .58
    location_effect = pd.Series(location).map({
        "Downtown": 5.0, "Riverside": 1.5, "West End": 3.0,
        "Midtown": 0.0, "Harbor": -3.5,
    }).to_numpy()
    peak_effect = np.select([arrival_hour == 19, arrival_hour == 20], [5.0, 3.0], default=0.0)
    waits = 8 + location_effect + .9 * party_size + peak_effect - 2.0 * reservation
    waits = np.clip(waits + rng.normal(0, 4.0, size), 0, None).round(1)
    fridays = pd.date_range("2026-01-02", periods=13, freq="7D")
    frame = pd.DataFrame({
        "observation_id": np.arange(1, size + 1),
        "date": rng.choice(fridays, size=size),
        "location_id": location,
        "wait_minutes": waits,
        "party_size": party_size,
        "reservation": reservation,
        "arrival_hour": arrival_hour,
        "service_period": np.where(arrival_hour <= 18, "early", np.where(arrival_hour == 19, "peak", "late")),
    })
    return frame.loc[:, REQUIRED_FIELDS]


def random_sample(population: pd.DataFrame, *, n: int = 40, seed: int = 21) -> pd.DataFrame:
    """Draw a reproducible simple random sample without replacement."""
    if not isinstance(n, int) or n <= 0 or n > len(population):
        raise ValueError("n must be a positive integer no larger than the population")
    return population.sample(n=n, replace=False, random_state=seed).copy().reset_index(drop=True)


def sample_statistics(sample: pd.DataFrame) -> pd.Series:
    """Calculate the observable mean, sample SD, and over-20 proportion."""
    if len(sample) < 2 or "wait_minutes" not in sample:
        raise ValueError("sample must contain at least two wait-time observations")
    waits = sample["wait_minutes"]
    return pd.Series({"mean": waits.mean(), "standard_deviation": waits.std(ddof=1),
                      "proportion_over_20": waits.gt(20).mean()})


def population_parameters(population: pd.DataFrame) -> pd.Series:
    """Reveal parameters only because this is a synthetic teaching population."""
    waits = population["wait_minutes"]
    return pd.Series({"mean": waits.mean(), "standard_deviation": waits.std(ddof=0),
                      "proportion_over_20": waits.gt(20).mean()})


def repeated_sample_means(population: pd.DataFrame, *, n: int = 40,
                          seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Draw only a handful of samples; Chapter 22 handles formal distributions."""
    return pd.DataFrame({"sample": range(1, len(seeds) + 1), "n": n,
                         "mean_wait": [random_sample(population, n=n, seed=s).wait_minutes.mean()
                                       for s in seeds]})


def sample_size_experiment(population: pd.DataFrame,
                           sizes: tuple[int, ...] = (10, 40, 200),
                           seeds: tuple[int, ...] = (11, 12, 13, 14, 15)) -> pd.DataFrame:
    """Compare a small handful of means at three sample sizes."""
    rows = []
    for n in sizes:
        for index, seed in enumerate(seeds, 1):
            rows.append({"n": n, "sample": index,
                         "mean_wait": random_sample(population, n=n, seed=seed).wait_minutes.mean()})
    return pd.DataFrame(rows)


def biased_sample(population: pd.DataFrame, *, n: int = 500, seed: int = 21) -> pd.DataFrame:
    """Mimic convenient collection at only the low-wait Harbor location."""
    eligible = population.query("location_id == 'Harbor'")
    if n > len(eligible):
        raise ValueError("not enough observations in the deliberately restricted subset")
    return random_sample(eligible, n=n, seed=seed)


def composition_summary(population: pd.DataFrame, random: pd.DataFrame,
                        biased: pd.DataFrame) -> pd.DataFrame:
    """Compare location shares while retaining absent locations as zeros."""
    locations = sorted(population.location_id.unique())
    result = pd.DataFrame({
        "population": population.location_id.value_counts(normalize=True).reindex(locations, fill_value=0),
        "random_sample": random.location_id.value_counts(normalize=True).reindex(locations, fill_value=0),
        "biased_sample": biased.location_id.value_counts(normalize=True).reindex(locations, fill_value=0),
    })
    return result.rename_axis("location_id").reset_index()


def stratified_sample(population: pd.DataFrame, *, per_location: int = 10,
                      seed: int = 21) -> pd.DataFrame:
    """Ensure each location appears, as intuition rather than weighted inference."""
    if per_location <= 0:
        raise ValueError("per_location must be positive")
    parts = [group.sample(n=per_location, random_state=seed + i)
             for i, (_, group) in enumerate(population.groupby("location_id", sort=True))]
    return pd.concat(parts, ignore_index=True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_figures(population: pd.DataFrame, random: pd.DataFrame, biased: pd.DataFrame,
                   repeated: pd.DataFrame, sizes: pd.DataFrame,
                   output_dir: Path) -> list[Path]:
    """Create five figures without introducing formal sampling distributions."""
    output_dir = Path(output_dir); paths = []; mu = population.wait_minutes.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axes[0].hist(population.wait_minutes, bins=25, color="#4c78a8"); axes[0].set_title("Synthetic population (N=5,000)")
    axes[1].hist(random.wait_minutes, bins=12, color="#f58518"); axes[1].set_title("One random sample (n=40)")
    for ax in axes: ax.set(xlabel="Wait (minutes)", ylabel="Parties")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.scatter(repeated["sample"], repeated.mean_wait, s=65); ax.axhline(mu, ls="--", color="black", label="population mean")
    ax.set(title="Five samples tell five slightly different stories", xlabel="Sample", ylabel="Sample mean (minutes)"); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))
    fig, ax = plt.subplots(figsize=(8, 4));
    for n, group in sizes.groupby("n"): ax.scatter([n] * len(group), group.mean_wait, label=f"n={n}", s=55)
    ax.axhline(mu, ls="--", color="black"); ax.set(xscale="log", xticks=[10, 40, 200], xticklabels=["10", "40", "200"], title="Larger random samples tend to fluctuate less", xlabel="Sample size", ylabel="Sample mean (minutes)"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))
    values = [mu, random.wait_minutes.mean(), biased.wait_minutes.mean()]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(["Population\nparameter", "Random\nn=40", "Biased\nn=500"], values, color=["#777777", "#4c78a8", "#e45756"]); ax.set(title="Size does not repair selection bias", ylabel="Mean wait (minutes)"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))
    comp = composition_summary(population, random, biased).set_index("location_id")
    fig, ax = plt.subplots(figsize=(10, 4)); comp.plot.bar(ax=ax); ax.set(title="Composition can reveal obvious coverage problems", xlabel="Location", ylabel="Share"); ax.legend(["Population", "Random n=40", "Biased n=500"]); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise Chapter 21 restaurant sampling experiment."""
    population = generate_population(); params = population_parameters(population)
    sample = random_sample(population); stats = sample_statistics(sample)
    repeated = repeated_sample_means(population); sizes = sample_size_experiment(population)
    biased = biased_sample(population); biased_stats = sample_statistics(biased)
    composition = composition_summary(population, sample, biased)
    stratified = stratified_sample(population)
    paths = create_figures(population, sample, biased, repeated, sizes,
                           output_dir or PROJECT_ROOT / "figures")
    print("Chapter 21 — Samples Tell Stories")
    print("Probability: Model → Data | Statistics: Data → Model / Population")
    print("Target population: All seated Friday dinner parties at James River Restaurant Group's five locations during Q1 2026.")
    print("Observational unit / grain: one seated party wait-time observation.")
    print(f"Synthetic teaching population: N={len(population):,}; fields={', '.join(REQUIRED_FIELDS)}")
    print("Population parameters (known only because this is a synthetic teaching population):")
    print(f"  μ={params['mean']:.2f} min; σ={params['standard_deviation']:.2f} min; p(wait>20)={params['proportion_over_20']:.1%}")
    print("Random sample n=40:")
    print(f"  Sample mean x̄={stats['mean']:.2f}; sample SD s={stats['standard_deviation']:.2f}; sample proportion p̂(wait>20)={stats['proportion_over_20']:.1%}")
    print(f"  Estimation difference x̄−μ={stats['mean'] - params['mean']:+.2f} minutes (one sample's error).")
    print("Five random-sample means: " + ", ".join(f"{x:.2f}" for x in repeated.mean_wait))
    ranges = sizes.groupby("n").mean_wait.agg(lambda x: x.max() - x.min())
    print("Sample-size intuition (range across five means): " + ", ".join(f"n={n}: {value:.2f}" for n, value in ranges.items()))
    print(f"Large biased sample: n={len(biased)} from Harbor only; mean={biased_stats['mean']:.2f}, difference from μ={biased_stats['mean'] - params['mean']:+.2f}.")
    print("Location composition shares (population / random / biased):\n" + composition.round(3).to_string(index=False))
    print(f"Stratified-sampling intuition: {len(stratified)} parties, 10 from each of five locations (unweighted teaching example).")
    print("Key lesson: Random variation changes statistics between samples; selection bias systematically studies the wrong subset. More rows do not fix bias or flawed measurement.")
    print("Generalization is defensible only with a relevant target, appropriate selection, reliable measurement, adequate information, awareness of dependence, and a stable process.")
    print("These observational data describe association, not causation. Rows are not automatically independent.")
    print(f"Generated {len(paths)} figures. Next question—not answered here: is there a predictable pattern to how x̄ varies across repeated samples of n=40?")
    return 0
