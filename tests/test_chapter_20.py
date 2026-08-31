"""Invariant, numerical, visual, and CLI tests for Chapter 20."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_foundations.chapter_20 import (
    BASELINE, PROMOTION, REQUIRED_FIGURES, StrategyAssumptions, break_even_customers,
    conditional_loss_probability, create_figures, decision_summary, incremental_analysis,
    sensitivity_analysis, simulate_strategy, stability_experiment,
)
from analytics_foundations.chapters import get_chapter


@pytest.mark.parametrize("assumptions,message", [
    (replace(PROMOTION, demand_probabilities=(.1, .2)), "equal nonzero"),
    (replace(PROMOTION, demand_probabilities=(.1, .2, .3, .2, .1)), "sum to 1"),
    (replace(PROMOTION, redemption_probability=1.1), "within"),
    (replace(PROMOTION, food_cost_rate=-.1), "within"),
    (replace(PROMOTION, average_spend_sd=-1), "nonnegative"),
])
def test_assumption_and_pmf_validation(assumptions, message):
    with pytest.raises(ValueError, match=message): assumptions.validate()


def test_reproducibility_shape_support_and_constraints():
    first = simulate_strategy(rng=np.random.default_rng(7), n_simulations=500, assumptions=PROMOTION)
    second = simulate_strategy(rng=np.random.default_rng(7), n_simulations=500, assumptions=PROMOTION)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 500 and set(first.customers) <= set(PROMOTION.demand_values)
    assert (first.redeemed_offers >= 0).all() and (first.redeemed_offers <= first.customers).all()
    assert (first.labor_hours >= PROMOTION.minimum_labor_hours).all()


def test_vectorized_costs_and_accounting_identity():
    frame = simulate_strategy(rng=np.random.default_rng(11), n_simulations=200, assumptions=PROMOTION)
    assert np.allclose(frame.revenue, frame.customers * frame.average_spend)
    assert np.allclose(frame.food_cost, PROMOTION.food_cost_rate * frame.revenue)
    assert np.allclose(frame.labor_cost, PROMOTION.hourly_labor_cost * frame.labor_hours)
    assert np.allclose(frame.promotion_cost, PROMOTION.fixed_promotion_cost + PROMOTION.discount_cost * frame.redeemed_offers)
    assert np.allclose(frame.profit, frame.revenue-frame.food_cost-frame.labor_cost-frame.promotion_cost)


def test_decision_summary_matches_numpy_and_probabilities_are_valid():
    frame = pd.DataFrame({"profit": [-100., 0., 100., 1000., 2000.]})
    summary = decision_summary(frame, target=500)
    assert summary.mean_profit == np.mean(frame.profit)
    assert summary.median_profit == np.median(frame.profit)
    assert summary.standard_deviation == pytest.approx(np.std(frame.profit, ddof=1))
    assert summary.p05 <= summary.median_profit <= summary.p95
    assert summary.probability_loss == .2 and summary.probability_above_target == .4
    assert 0 <= summary.probability_loss <= 1 and 0 <= summary.probability_above_target <= 1


def test_incremental_and_conditional_probability():
    baseline = pd.DataFrame({"profit": [0., 20., 30.], "customers": [120, 180, 120]})
    promotion = pd.DataFrame({"profit": [10., 10., 50.], "customers": [120, 180, 120]})
    result = incremental_analysis(baseline, promotion)
    assert np.array_equal(result.incremental_profit, promotion.profit-baseline.profit)
    assert (result.incremental_profit > 0).mean() == pytest.approx(2/3)
    risk = pd.DataFrame({"customers": [120, 120, 180], "profit": [-1., 2., -1.]})
    assert conditional_loss_probability(risk) == .5


def test_linked_dependence_and_independence_differ_reproducibly():
    linked = simulate_strategy(rng=np.random.default_rng(9), n_simulations=3000, assumptions=PROMOTION)
    independent = simulate_strategy(rng=np.random.default_rng(9), n_simulations=3000, assumptions=PROMOTION, linked_labor=False)
    assert linked.customers.corr(linked.labor_hours) > .7
    assert abs(independent.customers.corr(independent.labor_hours)) < .08
    assert not np.allclose(linked.profit, independent.profit)


def test_sensitivity_stability_and_break_even_are_deterministic():
    first = sensitivity_analysis(seed=3, n_simulations=300)
    second = sensitivity_analysis(seed=3, n_simulations=300)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 15 and set(first.level) == {"low", "base", "high"}
    stability = stability_experiment(seed=3, sizes=(20, 50))
    assert stability.n_simulations.tolist() == [20, 50]
    assert np.isfinite(stability[["mean_profit", "probability_loss"]]).all().all()
    assert break_even_customers() >= 0


def test_figures(tmp_path: Path):
    baseline = simulate_strategy(rng=np.random.default_rng(1), n_simulations=100, assumptions=BASELINE)
    promotion = simulate_strategy(rng=np.random.default_rng(1), n_simulations=100, assumptions=PROMOTION)
    paths = create_figures(promotion, baseline, incremental_analysis(baseline, promotion),
                           sensitivity_analysis(n_simulations=100),
                           stability_experiment(sizes=(20, 50)), tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-20")
    assert chapter and chapter.available and chapter.title == "Monte Carlo Business"
    assert chapter.run is not None
    # Exercise the public run with a temporary output directory.
    from analytics_foundations.chapter_20 import run
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Recommendation:" in output and "cannot conclude" in output
    assert len(output.splitlines()) < 35
