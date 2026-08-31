"""Monte Carlo restaurant-promotion decision model for Chapter 20."""

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT

REQUIRED_FIGURES = (
    "chapter-20-pipeline.png", "chapter-20-profit-histogram.png",
    "chapter-20-profit-ecdf.png", "chapter-20-strategy-comparison.png",
    "chapter-20-incremental-profit.png", "chapter-20-sensitivity.png",
    "chapter-20-stability.png",
)


@dataclass(frozen=True)
class StrategyAssumptions:
    """Explicit fixed parameters and probability-model parameters for one strategy."""

    name: str
    demand_values: tuple[int, ...]
    demand_probabilities: tuple[float, ...]
    average_spend_mean: float = 22.0
    average_spend_sd: float = 2.5
    minimum_spend: float = 1.0
    redemption_probability: float = 0.0
    food_cost_rate: float = 0.30
    hourly_labor_cost: float = 18.0
    base_labor_hours: float = 20.0
    labor_hours_per_customer: float = 0.12
    labor_noise_sd: float = 2.0
    minimum_labor_hours: float = 20.0
    fixed_promotion_cost: float = 0.0
    discount_cost: float = 0.0

    def validate(self) -> None:
        """Reject impossible or internally inconsistent model assumptions."""
        values = np.asarray(self.demand_values, dtype=float)
        probabilities = np.asarray(self.demand_probabilities, dtype=float)
        if len(values) == 0 or len(values) != len(probabilities):
            raise ValueError("demand values and probabilities must have equal nonzero length")
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("demand support must contain nonnegative integers")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("demand probabilities must be within [0, 1]")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("demand probabilities must sum to 1")
        rates = (self.redemption_probability, self.food_cost_rate)
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in rates):
            raise ValueError("redemption probability and food-cost rate must be within [0, 1]")
        nonnegative = (self.average_spend_sd, self.labor_noise_sd, self.hourly_labor_cost,
                       self.base_labor_hours, self.labor_hours_per_customer,
                       self.minimum_labor_hours, self.fixed_promotion_cost, self.discount_cost)
        if any(not np.isfinite(x) or x < 0 for x in nonnegative) or self.minimum_spend <= 0:
            raise ValueError("costs, scales, hours, and safeguards must be finite and nonnegative")
        if not np.isfinite(self.average_spend_mean):
            raise ValueError("average spend mean must be finite")


BASELINE = StrategyAssumptions(
    "No promotion", (90, 120, 150, 180, 210), (.10, .25, .35, .20, .10)
)
PROMOTION = StrategyAssumptions(
    "Friday promotion", (120, 150, 180, 210, 240), (.10, .20, .35, .25, .10),
    redemption_probability=.20, fixed_promotion_cost=300.0, discount_cost=4.0,
)


def deterministic_profit(assumptions: StrategyAssumptions, *, customers: int = 180,
                         average_spend: float = 22.0, labor_hours: float = 45.0,
                         redeemed_offers: int | None = None) -> dict[str, float]:
    """Calculate one transparent expected-input forecast or hand-worked evening."""
    assumptions.validate()
    redeemed = (round(customers * assumptions.redemption_probability)
                if redeemed_offers is None else redeemed_offers)
    revenue = customers * average_spend
    food_cost = assumptions.food_cost_rate * revenue
    labor_cost = labor_hours * assumptions.hourly_labor_cost
    promotion_cost = assumptions.fixed_promotion_cost + redeemed * assumptions.discount_cost
    return {"revenue": revenue, "food_cost": food_cost, "labor_cost": labor_cost,
            "promotion_cost": promotion_cost,
            "profit": revenue - food_cost - labor_cost - promotion_cost}


def simulate_strategy(*, rng: np.random.Generator, n_simulations: int,
                      assumptions: StrategyAssumptions,
                      linked_labor: bool = True) -> pd.DataFrame:
    """Vectorize plausible evenings using only the caller's explicit Generator."""
    assumptions.validate()
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    customers = rng.choice(assumptions.demand_values, n_simulations,
                           p=assumptions.demand_probabilities)
    spend = np.maximum(rng.normal(assumptions.average_spend_mean,
                                  assumptions.average_spend_sd, n_simulations),
                       assumptions.minimum_spend)
    redeemed = rng.binomial(customers, assumptions.redemption_probability)
    noise = rng.normal(0, assumptions.labor_noise_sd, n_simulations)
    if linked_labor:
        labor = assumptions.base_labor_hours + assumptions.labor_hours_per_customer * customers + noise
    else:
        expected_customers = np.dot(assumptions.demand_values, assumptions.demand_probabilities)
        labor = assumptions.base_labor_hours + assumptions.labor_hours_per_customer * expected_customers + noise
    labor = np.maximum(labor, assumptions.minimum_labor_hours)
    revenue = customers * spend
    food = assumptions.food_cost_rate * revenue
    labor_cost = labor * assumptions.hourly_labor_cost
    promo_cost = assumptions.fixed_promotion_cost + redeemed * assumptions.discount_cost
    profit = revenue - food - labor_cost - promo_cost
    frame = pd.DataFrame({"simulation": np.arange(1, n_simulations + 1), "customers": customers,
        "average_spend": spend, "redeemed_offers": redeemed, "labor_hours": labor,
        "revenue": revenue, "food_cost": food, "labor_cost": labor_cost,
        "promotion_cost": promo_cost, "profit": profit})
    validate_simulation(frame, assumptions)
    return frame


def validate_simulation(frame: pd.DataFrame, assumptions: StrategyAssumptions) -> None:
    """Reconcile simulated business rows as an analytical data pipeline."""
    required = {"customers", "average_spend", "redeemed_offers", "labor_hours", "revenue",
                "food_cost", "labor_cost", "promotion_cost", "profit"}
    if not required.issubset(frame) or not np.isfinite(frame[list(required)]).all().all():
        raise ValueError("simulation must contain required finite values")
    if not frame.customers.isin(assumptions.demand_values).all():
        raise ValueError("customers outside demand support")
    if ((frame.redeemed_offers < 0) | (frame.redeemed_offers > frame.customers)).any():
        raise ValueError("redemptions must be between zero and customers")
    if (frame.labor_hours < assumptions.minimum_labor_hours).any():
        raise ValueError("labor hours violate their lower bound")
    reconciled = frame.revenue - frame.food_cost - frame.labor_cost - frame.promotion_cost
    if not np.allclose(frame.profit, reconciled):
        raise ValueError("profit accounting identity does not reconcile")


def decision_summary(frame: pd.DataFrame, target: float = 1000.0) -> pd.Series:
    """Return decision-focused center, spread, percentile, and risk metrics."""
    profit = frame["profit"].to_numpy(dtype=float)
    if len(profit) == 0 or not np.isfinite(profit).all():
        raise ValueError("profit must contain finite observations")
    p05, median, p95 = np.percentile(profit, [5, 50, 95])
    return pd.Series({"mean_profit": profit.mean(), "median_profit": median,
        "standard_deviation": profit.std(ddof=1) if len(profit) > 1 else 0.0,
        "p05": p05, "p95": p95, "probability_loss": np.mean(profit < 0),
        "probability_above_target": np.mean(profit > target)})


def incremental_analysis(baseline: pd.DataFrame, promotion: pd.DataFrame) -> pd.DataFrame:
    """Align trials and calculate promotion profit minus baseline profit."""
    if len(baseline) != len(promotion):
        raise ValueError("strategy simulations must have equal lengths")
    return pd.DataFrame({"simulation": np.arange(1, len(baseline) + 1),
        "baseline_profit": baseline.profit.to_numpy(), "promotion_profit": promotion.profit.to_numpy(),
        "incremental_profit": promotion.profit.to_numpy() - baseline.profit.to_numpy()})


def conditional_loss_probability(frame: pd.DataFrame, customer_threshold: int = 150) -> float:
    """Estimate P(loss | customers below threshold), requiring a nonempty event."""
    low = frame.customers < customer_threshold
    if not low.any():
        raise ValueError("conditioning event has no simulated observations")
    return float((frame.loc[low, "profit"] < 0).mean())


def sensitivity_analysis(assumptions: StrategyAssumptions = PROMOTION, *, seed: int = 2020,
                         n_simulations: int = 3000) -> pd.DataFrame:
    """Run reproducible low/base/high one-at-a-time assumption changes."""
    settings = {
        "redemption_probability": (.10, assumptions.redemption_probability, .30),
        "average_spend_mean": (20.0, assumptions.average_spend_mean, 24.0),
        "food_cost_rate": (.27, assumptions.food_cost_rate, .33),
        "labor_hours_per_customer": (.10, assumptions.labor_hours_per_customer, .14),
        "fixed_promotion_cost": (200.0, assumptions.fixed_promotion_cost, 400.0),
    }
    rows = []
    for parameter, values in settings.items():
        for level, value in zip(("low", "base", "high"), values, strict=True):
            result = simulate_strategy(rng=np.random.default_rng(seed), n_simulations=n_simulations,
                                       assumptions=replace(assumptions, **{parameter: value}))
            summary = decision_summary(result)
            rows.append({"assumption": parameter, "level": level, "value": value,
                         "expected_profit": summary.mean_profit,
                         "probability_loss": summary.probability_loss})
    return pd.DataFrame(rows)


def stability_experiment(assumptions: StrategyAssumptions = PROMOTION, *, seed: int = 2020,
                         sizes: tuple[int, ...] = (100, 1000, 10000)) -> pd.DataFrame:
    """Compare finite-simulation estimates without claiming monotonic improvement."""
    rows = []
    for size in sizes:
        summary = decision_summary(simulate_strategy(rng=np.random.default_rng(seed),
                                    n_simulations=size, assumptions=assumptions))
        rows.append({"n_simulations": size, "mean_profit": summary.mean_profit,
                     "probability_loss": summary.probability_loss})
    return pd.DataFrame(rows)


def break_even_customers(assumptions: StrategyAssumptions = PROMOTION) -> int:
    """Find the fixed-input customer threshold yielding nonnegative expected profit."""
    for customers in range(1000):
        labor = assumptions.base_labor_hours + assumptions.labor_hours_per_customer * customers
        if deterministic_profit(assumptions, customers=customers,
                average_spend=assumptions.average_spend_mean, labor_hours=labor)["profit"] >= 0:
            return customers
    raise ValueError("no break-even point below 1,000 customers")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); return path


def create_figures(promotion: pd.DataFrame, baseline: pd.DataFrame, incremental: pd.DataFrame,
                   sensitivity: pd.DataFrame, stability: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate seven transparent Monte Carlo teaching figures."""
    output_dir = Path(output_dir); paths = []
    fig, ax = plt.subplots(figsize=(9, 4)); ax.axis("off")
    ax.text(.02, .65, "Demand   Spend   Redemptions   Labor   Costs", fontsize=12)
    ax.annotate("Business model  →  Profit", xy=(.95, .65), xytext=(.43, .65), arrowprops={"arrowstyle": "->"}, fontsize=13)
    ax.text(.35, .25, "repeat thousands of times  ↓  profit distribution", fontsize=12)
    paths.append(_save(fig, output_dir / REQUIRED_FIGURES[0]))
    s = decision_summary(promotion); fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(promotion.profit, bins=45, color="#4c78a8", alpha=.8)
    for x, label, color in [(0, "zero", "black"), (s.mean_profit, "mean", "#e45756"), (s.median_profit, "median", "#72b7b2"), (s.p05, "5th percentile", "#f58518")]: ax.axvline(x, label=label, color=color, ls="--")
    ax.set(title="Promotion profit across simulated evenings", xlabel="Profit ($)", ylabel="Evenings"); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[1]))
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.sort(promotion.profit); ax.plot(x, np.arange(1, len(x)+1)/len(x)); ax.axvline(0, color="black", ls="--", label="zero"); ax.axvline(1000, color="#e45756", ls=":", label="$1,000 target"); ax.set(title="Empirical CDF of promotion profit", xlabel="Profit ($)", ylabel="P(Profit ≤ x)"); ax.legend(); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[2]))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.boxplot([baseline.profit, promotion.profit], tick_labels=["No promotion", "Promotion"], showfliers=False); ax.axhline(0, color="black", ls="--"); ax.set(title="Comparable strategy profit distributions", ylabel="Profit ($)"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[3]))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(incremental.incremental_profit, bins=45, color="#54a24b"); ax.axvline(0, color="black", ls="--"); ax.set(title="Incremental profit: promotion minus baseline", xlabel="Incremental profit ($)", ylabel="Evenings"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[4]))
    pivot = sensitivity.pivot(index="assumption", columns="level", values="expected_profit")[["low", "base", "high"]]; fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.barh(ax=ax); ax.set(title="One-at-a-time sensitivity", xlabel="Expected profit ($)", ylabel="Assumption"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[5]))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(stability.n_simulations, stability.mean_profit, marker="o"); ax.set_xscale("log"); ax.set(title="Finite-simulation estimates become more stable", xlabel="Number of simulations (log scale)", ylabel="Estimated mean profit ($)"); paths.append(_save(fig, output_dir / REQUIRED_FIGURES[6])); return paths


def run(output_dir: Path | None = None) -> int:
    """Run the concise, decision-oriented Chapter 20 experiment."""
    n, seed = 10_000, 2020
    forecast = deterministic_profit(PROMOTION)
    one = deterministic_profit(PROMOTION, customers=210, average_spend=21.40, labor_hours=47.2, redeemed_offers=39)
    baseline = simulate_strategy(rng=np.random.default_rng(seed), n_simulations=n, assumptions=BASELINE)
    promotion = simulate_strategy(rng=np.random.default_rng(seed), n_simulations=n, assumptions=PROMOTION)
    independent = simulate_strategy(rng=np.random.default_rng(seed), n_simulations=n, assumptions=PROMOTION, linked_labor=False)
    bs, ps, ins = decision_summary(baseline), decision_summary(promotion), decision_summary(independent)
    incremental = incremental_analysis(baseline, promotion); win = (incremental.incremental_profit > 0).mean()
    sensitivity = sensitivity_analysis(); stability = stability_experiment()
    correlations = promotion[["customers", "labor_hours", "revenue", "profit"]].corr()
    paths = create_figures(promotion, baseline, incremental, sensitivity, stability, output_dir or PROJECT_ROOT / "figures")
    most_sensitive = (sensitivity.groupby("assumption").expected_profit.agg(lambda x: x.max()-x.min()).idxmax())
    recommend = "test the promotion with a controlled pilot" if win > .5 else "retain the baseline"
    print("Chapter 20 — Monte Carlo Business")
    print("Decision: James River Restaurant Group is evaluating one Friday promotion at one location.")
    print(f"Deterministic expected-input forecast: customers=180, spend=$22, labor=45h, redemptions=36 → profit=${forecast['profit']:,.2f}. This single number hides combinations of demand, spend, redemption, and labor.")
    print("Inputs: discrete promotion demand {120,…,240}; spend Normal(22, 2.5²), clipped at $1; redemptions | customers ~ Binomial(customers, .20); labor=20+.12(customers)+Normal(0,2²), minimum 20h. Food rate, labor rate, and setup cost are fixed.")
    print(f"One plausible evening—not a forecast: 210 customers, $21.40 spend, 39 redemptions, 47.2 labor hours → revenue=${one['revenue']:,.2f}, food=${one['food_cost']:,.2f}, labor=${one['labor_cost']:,.2f}, promotion=${one['promotion_cost']:,.2f}, profit=${one['profit']:,.2f}.")
    summary = pd.DataFrame({"baseline": bs, "promotion": ps}).T
    print("10,000-evening decision summary:\n" + summary.round(3).to_string())
    print(f"Incremental profit: mean=${incremental.incremental_profit.mean():,.2f}; P(promotion > baseline)={win:.1%}. P(promotion loss | customers<150)={conditional_loss_probability(promotion):.1%}; unconditional={ps.probability_loss:.1%}.")
    print(f"Dependence: corr(customers,labor)={correlations.loc['customers','labor_hours']:.3f}, corr(customers,revenue)={correlations.loc['customers','revenue']:.3f}, corr(labor,profit)={correlations.loc['labor_hours','profit']:.3f}. Independent-labor profit SD=${ins.standard_deviation:,.2f} versus linked=${ps.standard_deviation:,.2f}; convenience changes risk.")
    print(f"Fixed-input break-even is {break_even_customers()} customers, but uncertain spend and redemption make that threshold insufficient. Most decision-sensitive tested assumption: {most_sensitive}.")
    print("Stability (estimates need not improve monotonically):\n" + stability.round(3).to_string(index=False))
    print(f"Recommendation: {recommend}. Evidence: expected promotion profit=${ps.mean_profit:,.0f}, loss risk={ps.probability_loss:.1%}, 5th percentile=${ps.p05:,.0f}, and scenario win rate={win:.1%}. Validate {most_sensitive}, demand support, redemption, and demand-linked staffing with observed data before scaling.")
    print("Limitations / cannot conclude: simulation does not prove profitability or a causal promotion effect; simulated demand is not observed future demand; probabilities are conditional on fixed, possibly uncertain parameters; omitted dependencies, stale data, tail behavior, structural change, and bugs create model error. More trials reduce simulation error—not model or parameter uncertainty.")
    print(f"Generated {len(paths)} figures. Same seed + code + assumptions reproduces results; different seeds change individual evenings, not the intended large-run behavior.")
    return 0
