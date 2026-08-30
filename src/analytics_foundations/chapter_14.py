"""Events, probability models, and simulation for Chapter 14."""

from collections.abc import Mapping, Set
from numbers import Real
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from numpy.typing import NDArray

from analytics_foundations.datasets import PROJECT_ROOT

DEMAND_MODEL = {
    "Low": 0.15,
    "Moderate": 0.35,
    "Busy": 0.30,
    "Very Busy": 0.20,
}
BUSY_OR_HIGHER = {"Busy", "Very Busy"}
REQUIRED_FIGURES = (
    "chapter-14-demand-probabilities.png",
    "chapter-14-event-in-sample-space.png",
    "chapter-14-complement.png",
    "chapter-14-union.png",
    "chapter-14-intersection.png",
    "chapter-14-simulation-convergence.png",
)


def validate_probability_model(
    probabilities: Mapping[str, Real], *, tolerance: float = 1e-9
) -> None:
    """Validate a finite categorical probability model, failing clearly."""
    if not probabilities:
        raise ValueError("a probability model must contain at least one outcome")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           for value in probabilities.values()):
        raise TypeError("every probability must be numeric")
    if any(not np.isfinite(value) or value < 0 or value > 1
           for value in probabilities.values()):
        raise ValueError("every probability must be between 0 and 1")
    if not np.isclose(sum(probabilities.values()), 1.0, atol=tolerance, rtol=0):
        raise ValueError("probabilities must sum to 1")


def probability_of_event(
    probabilities: Mapping[str, Real], event: Set[str]
) -> float:
    """Return the probability of an event represented as a set of outcomes."""
    validate_probability_model(probabilities)
    unknown = set(event) - set(probabilities)
    if unknown:
        raise ValueError(f"event contains outcomes outside the sample space: {sorted(unknown)}")
    return float(sum(probabilities[outcome] for outcome in event))


def complement_probability(
    probabilities: Mapping[str, Real], event: Set[str]
) -> float:
    """Return P(event complement), validating the event along the way."""
    return 1.0 - probability_of_event(probabilities, event)


def union_probability(
    probabilities: Mapping[str, Real], event_a: Set[str], event_b: Set[str]
) -> float:
    """Return P(A union B), whether or not the events overlap."""
    return probability_of_event(probabilities, set(event_a) | set(event_b))


def intersection_probability(
    probabilities: Mapping[str, Real], event_a: Set[str], event_b: Set[str]
) -> float:
    """Return P(A intersection B)."""
    return probability_of_event(probabilities, set(event_a) & set(event_b))


def empirical_probability(outcomes: NDArray[np.str_], event: Set[str]) -> float:
    """Estimate an event probability from nonempty observed outcomes."""
    if outcomes.size == 0:
        raise ValueError("empirical probability requires at least one observation")
    return float(np.mean(np.isin(outcomes, list(event))))


def simulate_categorical_outcomes(
    probabilities: Mapping[str, Real], size: int, *, seed: int
) -> NDArray[np.str_]:
    """Simulate categorical trials reproducibly with NumPy's Generator API."""
    validate_probability_model(probabilities)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.array(list(probabilities)), size=size,
        p=np.array(list(probabilities.values()), dtype=float),
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _set_axes(ax: plt.Axes, title: str) -> None:
    ax.set(xlim=(0, 10), ylim=(0, 6), aspect="equal", title=title)
    ax.axis("off")
    ax.add_patch(Rectangle((.3, .3), 9.4, 5.2, fill=False, linewidth=1.8))
    ax.text(.55, 5.05, r"$\Omega$", fontsize=13)


def _venn_figure(title: str, operation: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _set_axes(ax, title)
    left = Circle((4, 3), 1.7, edgecolor="#24557a", linewidth=2)
    right = Circle((6, 3), 1.7, edgecolor="#8b4513", linewidth=2)
    for circle in (left, right):
        ax.add_patch(circle)
    if operation == "union":
        left.set(facecolor="#72b7b2", alpha=.55)
        right.set(facecolor="#72b7b2", alpha=.55)
    elif operation == "intersection":
        left.set(facecolor="#dddddd", alpha=.25)
        right.set(facecolor="#dddddd", alpha=.25)
        lens = Circle((5, 3), .72, facecolor="#eeca3b", alpha=.9, edgecolor="none")
        lens.set_clip_path(left)
        ax.add_patch(lens)
        # Overlaid clipped circles make the shared region visually unmistakable.
        overlap = Circle((6, 3), 1.7, facecolor="#eeca3b", alpha=.75, edgecolor="none")
        overlap.set_clip_path(left)
        ax.add_patch(overlap)
    ax.text(3.25, 3, "A", fontsize=14)
    ax.text(6.55, 3, "B", fontsize=14)
    ax.text(5, .7, "OR: either or both" if operation == "union" else "AND: both",
            ha="center", weight="bold")
    return fig


def create_figures(output_dir: Path, *, seed: int = 14) -> list[Path]:
    """Create the chapter's deterministic probability and set visuals."""
    validate_probability_model(DEMAND_MODEL)
    paths: list[Path] = []
    states = list(DEMAND_MODEL)
    values = list(DEMAND_MODEL.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(states, values, color="#4c78a8")
    ax.bar_label(bars, fmt="%.2f")
    ax.set(title="Tomorrow's Demand Probability Model", xlabel="Demand state",
           ylabel="Probability", ylim=(0, .42))
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    colors = ["#d9d9d9" if state not in BUSY_OR_HIGHER else "#f28e2b" for state in states]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(states, values, color=colors)
    ax.set(title=r"Event $A=\{Busy, Very\ Busy\}$: at least busy",
           xlabel="Outcomes in the sample space", ylabel="Probability", ylim=(0, .42))
    ax.text(2.5, .36, r"Highlighted event: $P(A)=0.50$", ha="center")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _set_axes(ax, r"Complement: everything in $\Omega$ outside $A$")
    ax.add_patch(Rectangle((.3, .3), 9.4, 5.2, color="#72b7b2", alpha=.45))
    ax.add_patch(Circle((5, 3), 1.55, facecolor="white", edgecolor="#24557a", linewidth=2))
    ax.text(5, 3, "A\nBusy or\nVery Busy", ha="center", va="center")
    ax.text(7.9, 1.2, r"$A^c$: Low or Moderate", ha="center", weight="bold")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    paths.append(_save(_venn_figure(r"Union $A\cup B$", "union"), output_dir / REQUIRED_FIGURES[3]))
    paths.append(_save(_venn_figure(r"Intersection $A\cap B$", "intersection"), output_dir / REQUIRED_FIGURES[4]))

    simulated = simulate_categorical_outcomes(DEMAND_MODEL, 10_000, seed=seed)
    indicators = np.isin(simulated, list(BUSY_OR_HIGHER))
    cumulative = np.cumsum(indicators) / np.arange(1, indicators.size + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, indicators.size + 1), cumulative, linewidth=1,
            label="Cumulative empirical proportion")
    ax.axhline(.5, color="#e45756", linestyle="--", label="Model P(A) = 0.50")
    ax.set(title="Simulation: Empirical Frequency Can Fluctuate",
           xlabel="Number of simulated days", ylabel="Busy-or-higher proportion",
           xscale="log", ylim=(0, 1))
    ax.legend()
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the demand probability experiment and print a concise interpretation."""
    validate_probability_model(DEMAND_MODEL)
    event_probability = probability_of_event(DEMAND_MODEL, BUSY_OR_HIGHER)
    die = {str(number): 1 / 6 for number in range(1, 7)}
    even, at_least_four = {"2", "4", "6"}, {"4", "5", "6"}
    print("Chapter 14 — Events & Probability")
    print("Question: How can we describe tomorrow's uncertain restaurant demand mathematically?")
    print("Observed: yesterday 184 customers visited. Uncertain: tomorrow may be Low, Moderate, Busy, or Very Busy.")
    print("Sample space Ω = {Low, Moderate, Busy, Very Busy}")
    print("Demand probability model (validated):")
    for outcome, probability in DEMAND_MODEL.items():
        print(f"  {outcome:<10} {probability:.2f}")
    print(f"P(Busy)={probability_of_event(DEMAND_MODEL, {'Busy'}):.2f}; P(Busy or Very Busy)={event_probability:.2f}; complement={complement_probability(DEMAND_MODEL, BUSY_OR_HIGHER):.2f}.")
    print(f"Mutually exclusive union P(Low or Very Busy)={union_probability(DEMAND_MODEL, {'Low'}, {'Very Busy'}):.2f}.")
    print(f"Fair-die overlap: A∪B={sorted(even | at_least_four)}, A∩B={sorted(even & at_least_four)}, P(A∪B)={union_probability(die, even, at_least_four):.3f}.")
    print("Simulation (fixed seed 14): " + ", ".join(
        f"n={size:,}: {empirical_probability(simulate_categorical_outcomes(DEMAND_MODEL, size, seed=14), BUSY_OR_HIGHER):.3f}"
        for size in (10, 100, 1_000, 10_000)))
    print("Small batches vary (10 trials): " + ", ".join(
        f"seed {seed}: {empirical_probability(simulate_categorical_outcomes(DEMAND_MODEL, 10, seed=seed), BUSY_OR_HIGHER):.0%}"
        for seed in (3, 7, 0)))
    paths = create_figures(output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures: distribution, event, complement, union, intersection, and convergence.")
    print("Interpretation: P(A)=0.50 informs staffing, but costs and consequences also matter.")
    print("Not guaranteed: half of every small batch will be busy-or-higher; simulation executes the supplied model, not the unknown truth.")
    return 0
