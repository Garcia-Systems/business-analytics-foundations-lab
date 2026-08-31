"""Conditional probability, independence, and Bayes' rule for Chapter 15."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT


REQUIRED_FIGURES = (
    "chapter-15-restricted-denominator.png",
    "chapter-15-contingency-table.png",
    "chapter-15-probability-tree.png",
    "chapter-15-bayes-counts.png",
)


@dataclass(frozen=True)
class FraudCounts:
    """Expected binary-screening counts for an integer-sized population."""

    fraud: int
    no_fraud: int
    true_alerts: int
    missed_fraud: int
    false_alerts: int
    true_negatives: int

    @property
    def total_alerts(self) -> int:
        return self.true_alerts + self.false_alerts

    @property
    def posterior(self) -> float:
        if self.total_alerts == 0:
            raise ValueError("posterior is undefined when there are no alerts")
        return self.true_alerts / self.total_alerts


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability between 0 and 1")
    return float(value)


def conditional_probability(
    joint_probability: float, condition_probability: float
) -> float:
    """Return P(A|B)=P(A intersection B)/P(B)."""
    joint = _probability(joint_probability, "joint_probability")
    condition = _probability(condition_probability, "condition_probability")
    if condition == 0:
        raise ValueError("condition_probability must be greater than zero")
    if joint > condition:
        raise ValueError("joint_probability cannot exceed condition_probability")
    return joint / condition


def are_independent(
    probability_a: float, probability_b: float, joint_probability: float,
    *, tolerance: float = 1e-9,
) -> bool:
    """Test independence using P(A intersection B)=P(A)P(B)."""
    a = _probability(probability_a, "probability_a")
    b = _probability(probability_b, "probability_b")
    joint = _probability(joint_probability, "joint_probability")
    if joint > min(a, b):
        raise ValueError("joint_probability cannot exceed either marginal probability")
    return bool(np.isclose(joint, a * b, atol=tolerance, rtol=0))


def bayes_binary(
    prior: float, sensitivity: float, false_positive_rate: float
) -> float:
    """Return P(target|alert) for a binary alert system."""
    prior = _probability(prior, "prior")
    sensitivity = _probability(sensitivity, "sensitivity")
    false_positive_rate = _probability(false_positive_rate, "false_positive_rate")
    numerator = sensitivity * prior
    alert_probability = numerator + false_positive_rate * (1 - prior)
    if alert_probability == 0:
        raise ValueError("the supplied model gives the conditioning event zero probability")
    return numerator / alert_probability


def fraud_count_table(
    population: int = 10_000, *, fraud_rate: float = .01,
    sensitivity: float = .90, false_positive_rate: float = .05,
) -> FraudCounts:
    """Convert a binary Bayes model into transparent expected integer counts."""
    if not isinstance(population, int) or isinstance(population, bool) or population <= 0:
        raise ValueError("population must be a positive integer")
    fraud_rate = _probability(fraud_rate, "fraud_rate")
    sensitivity = _probability(sensitivity, "sensitivity")
    false_positive_rate = _probability(false_positive_rate, "false_positive_rate")
    fraud = round(population * fraud_rate)
    no_fraud = population - fraud
    true_alerts = round(fraud * sensitivity)
    false_alerts = round(no_fraud * false_positive_rate)
    return FraudCounts(
        fraud, no_fraud, true_alerts, fraud - true_alerts,
        false_alerts, no_fraud - false_alerts,
    )


def simulate_fraud_alerts(
    n: int = 200_000, *, fraud_rate: float = .01, sensitivity: float = .90,
    false_positive_rate: float = .05, seed: int = 15,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Simulate fraud status, then alerts conditional on that status."""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    fraud_rate = _probability(fraud_rate, "fraud_rate")
    sensitivity = _probability(sensitivity, "sensitivity")
    false_positive_rate = _probability(false_positive_rate, "false_positive_rate")
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < fraud_rate
    alert = np.empty(n, dtype=bool)
    alert[fraud] = rng.random(int(fraud.sum())) < sensitivity
    alert[~fraud] = rng.random(int((~fraud).sum())) < false_positive_rate
    return fraud, alert


def build_day_dataset() -> pd.DataFrame:
    """Construct 100 day-level observations matching the teaching table."""
    friday_dates = pd.date_range("2025-01-03", periods=24, freq="7D")
    other_dates = pd.date_range("2024-01-01", periods=76, freq="D")
    other_dates = other_dates[other_dates.dayofweek != 4]
    while len(other_dates) < 76:
        last = other_dates[-1] + pd.Timedelta(days=1)
        additions = pd.date_range(last, periods=20, freq="D")
        other_dates = other_dates.append(additions[additions.dayofweek != 4])
    dates = friday_dates.append(other_dates[:76])
    is_friday = np.r_[np.ones(24, dtype=bool), np.zeros(76, dtype=bool)]
    busy = np.r_[np.ones(18, dtype=bool), np.zeros(6, dtype=bool),
                 np.ones(22, dtype=bool), np.zeros(54, dtype=bool)]
    promotion = np.zeros(100, dtype=bool)
    promotion[[0, 1, 20, 30, 61, 83]] = True  # three Friday promotions; two busy
    rain = np.array([(index * 7 + 3) % 11 < 3 for index in range(100)])
    return pd.DataFrame({
        "date": dates, "is_friday": is_friday, "busy": busy,
        "promotion_active": promotion, "rain": rain,
    }).sort_values("date", ignore_index=True)


def contingency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Friday-by-demand count table with pedagogical labels/totals."""
    table = pd.crosstab(df["is_friday"], df["busy"])
    table = table.reindex(index=[True, False], columns=[True, False], fill_value=0)
    table.index = ["Friday", "Not Friday"]
    table.columns = ["Busy", "Not Busy"]
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_figures(output_dir: Path) -> list[Path]:
    """Create four deterministic visuals for conditioning and Bayes reasoning."""
    paths: list[Path] = []
    df = build_day_dataset()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = np.where(df["is_friday"], "#f28e2b", "#d9d9d9")
    for ax, restricted in zip(axes, (False, True), strict=True):
        for index in range(100):
            visible = not restricted or bool(df.iloc[index]["is_friday"])
            ax.scatter(index % 10, 9 - index // 10, s=90,
                       color=colors[index] if visible else "white",
                       edgecolor="#777777" if visible else "#eeeeee")
        ax.set_title("All days: denominator = 100" if not restricted
                     else "Given Friday: denominator = 24")
        ax.axis("off")
    fig.suptitle("Conditioning restricts which observations remain relevant")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))

    counts = np.array([[18, 6], [22, 54]])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(counts, cmap="Blues")
    for (row, column), value in np.ndenumerate(counts):
        ax.text(column, row, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > 30 else "#222222")
    ax.set(xticks=[0, 1], xticklabels=["Busy", "Not Busy"],
           yticks=[0, 1], yticklabels=["Friday", "Not Friday"],
           title="Restaurant demand contingency table (counts)")
    fig.colorbar(image, ax=ax, label="Days")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    nodes = {"start": (0, .5), "F": (.34, .75), "NF": (.34, .25),
             "FB": (.8, .9), "FNB": (.8, .62), "NFB": (.8, .38), "NFNB": (.8, .1)}
    edges = [("start", "F", "P(F)=.24"), ("start", "NF", "P(Fᶜ)=.76"),
             ("F", "FB", "P(B|F)=.75"), ("F", "FNB", ".25"),
             ("NF", "NFB", "P(B|Fᶜ)=22/76"), ("NF", "NFNB", "54/76")]
    for source, target, label in edges:
        x1, y1 = nodes[source]; x2, y2 = nodes[target]
        ax.plot([x1, x2], [y1, y2], color="#4c78a8", linewidth=2)
        ax.text((x1+x2)/2, (y1+y2)/2+.025, label, ha="center", fontsize=9)
    for key, (x, y) in nodes.items():
        labels = {"start": "Day", "F": "Friday", "NF": "Not Friday",
                  "FB": "Busy\njoint=.18", "FNB": "Not Busy\njoint=.06",
                  "NFB": "Busy\njoint=.22", "NFNB": "Not Busy\njoint=.54"}
        ax.text(x, y, labels[key], ha="center", va="center",
                bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#555555"})
    ax.set_title("Multiply along a path to obtain a joint probability")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))

    counts_model = fraud_count_table()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    boxes = [("10,000 transactions", .04, .4, "#dddddd"),
             ("100 fraud", .35, .68, "#f28e2b"), ("9,900 not fraud", .35, .18, "#4c78a8"),
             ("90 true alerts", .69, .75, "#e45756"), ("10 missed", .69, .57, "#eeeeee"),
             ("495 false alerts", .69, .27, "#e45756"), ("9,405 no alert", .69, .09, "#eeeeee")]
    for label, x, y, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), .22, .11, boxstyle="round,pad=.02",
                                   facecolor=color, edgecolor="#555555"))
        ax.text(x+.11, y+.055, label, ha="center", va="center", fontsize=9)
    for y1, y2 in ((.455, .735), (.455, .235), (.735, .805), (.735, .625),
                   (.235, .325), (.235, .145)):
        x1, x2 = ((.26, .35) if y1 == .455 else (.57, .69))
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->"})
    ax.set_title(f"Bayes by counts: {counts_model.true_alerts} of {counts_model.total_alerts} alerts are fraud ({counts_model.posterior:.1%})")
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))
    return paths


def run(output_dir: Path | None = None) -> int:
    """Run the conditional-probability chapter experiment."""
    df = build_day_dataset()
    table = contingency_table(df)
    total = len(df)
    p_busy = table.loc["Total", "Busy"] / total
    p_friday = table.loc["Friday", "Total"] / total
    p_joint = table.loc["Friday", "Busy"] / total
    friday_rows = df[df["is_friday"]]
    p_busy_given_friday = float(friday_rows["busy"].mean())
    p_friday_given_busy = float(df.loc[df["busy"], "is_friday"].mean())
    fraud_counts = fraud_count_table()
    theoretical = bayes_binary(.01, .90, .05)
    fraud, alert = simulate_fraud_alerts()
    simulated = float(fraud[alert].mean())

    print("Chapter 15 — Conditional Probability")
    print("Question from Chapter 14: What is P(Busy | Friday)? The bar means 'given that'.")
    print("Dataset grain: one row per observed restaurant day (100 rows).")
    print("Friday × Busy count table:")
    print(table.to_string())
    print(f"Marginals: P(Busy)={p_busy:.2f}; P(Friday)={p_friday:.2f}. Joint P(Busy ∩ Friday)={p_joint:.2f}.")
    print(f"Conditioning changes the denominator: P(Busy | Friday)=18/24={p_busy_given_friday:.2f}; P(Friday | Busy)=18/40={p_friday_given_busy:.2f}.")
    print(f"Multiplication rule: {p_busy_given_friday:.2f} × {p_friday:.2f} = {p_busy_given_friday*p_friday:.2f} = P(Busy ∩ Friday).")
    print(f"Independence check: P(Busy | Friday)={p_busy_given_friday:.2f} differs from P(Busy)={p_busy:.2f}; not independent in these data.")
    print(f"Bayes by counts: 100 fraud, 90 true alerts; 9,900 non-fraud, 495 false alerts; P(Fraud | Alert)=90/{fraud_counts.total_alerts}={fraud_counts.posterior:.3f}.")
    print(f"Bayes formula={theoretical:.3f}; fixed-seed simulation={simulated:.3f} (n={len(fraud):,}).")
    paths = create_figures(output_dir or PROJECT_ROOT / "figures")
    print(f"Generated {len(paths)} figures: restricted denominator, contingency table, probability tree, and Bayes counts.")
    print("Business interpretation: Friday context can inform staffing, but costs, revenue, weather, promotions, reservations, and estimate uncertainty still matter.")
    print("Limitations: association is not causation; narrow conditions can leave sparse evidence; simulation cannot rescue incorrect assumptions.")
    return 0
