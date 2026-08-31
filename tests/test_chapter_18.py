"""Calculations, simulations, figures, and registration for Chapter 18."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_18 import (
    PROFIT_PROBABILITIES, PROFIT_VALUES, REQUIRED_FIGURES, contribution_table,
    create_figures, discrete_standard_deviation, discrete_variance, expected_value,
    run, simulate_profit, variance_shortcut,
)
from analytics_foundations.chapters import get_chapter


def test_expectation_contributions_and_pmf_validation_reuse():
    table = contribution_table(PROFIT_VALUES, PROFIT_PROBABILITIES)
    assert expected_value(PROFIT_VALUES, PROFIT_PROBABILITIES) == pytest.approx(1425)
    assert table.weighted_value.sum() == pytest.approx(1425)
    with pytest.raises(ValueError, match="sum to 1"):
        expected_value([1, 2], [.2, .2])


def test_variance_definition_shortcut_and_standard_deviation():
    variance = discrete_variance(PROFIT_VALUES, PROFIT_PROBABILITIES)
    assert variance == pytest.approx(2_606_875)
    assert variance_shortcut(PROFIT_VALUES, PROFIT_PROBABILITIES) == pytest.approx(variance)
    assert contribution_table(PROFIT_VALUES, PROFIT_PROBABILITIES).weighted_squared_deviation.sum() == pytest.approx(variance)
    assert discrete_standard_deviation(PROFIT_VALUES, PROFIT_PROBABILITIES) == pytest.approx(np.sqrt(variance))


def test_shift_scale_and_expectation_transformations():
    mu = expected_value(PROFIT_VALUES, PROFIT_PROBABILITIES)
    variance = discrete_variance(PROFIT_VALUES, PROFIT_PROBABILITIES)
    shifted = PROFIT_VALUES + 100
    scaled = 2 * PROFIT_VALUES
    assert expected_value(3 * PROFIT_VALUES + 5, PROFIT_PROBABILITIES) == pytest.approx(3 * mu + 5)
    assert discrete_variance(shifted, PROFIT_PROBABILITIES) == pytest.approx(variance)
    assert discrete_variance(scaled, PROFIT_PROBABILITIES) == pytest.approx(4 * variance)
    assert discrete_standard_deviation(scaled, PROFIT_PROBABILITIES) == pytest.approx(2 * np.sqrt(variance))


def test_named_distribution_moments_and_same_mean_different_spread():
    p, n = .2, 20
    assert expected_value([0, 1], [1-p, p]) == pytest.approx(p)
    assert discrete_variance([0, 1], [1-p, p]) == pytest.approx(p*(1-p))
    binomial_values = np.arange(n+1)
    from scipy.stats import binom
    probabilities = binom.pmf(binomial_values, n, p)
    assert expected_value(binomial_values, probabilities) == pytest.approx(n*p)
    assert discrete_variance(binomial_values, probabilities) == pytest.approx(n*p*(1-p))
    stable, risky, probs = [900, 1100], [-1000, 3000], [.5, .5]
    assert expected_value(stable, probs) == expected_value(risky, probs) == 1000
    assert discrete_standard_deviation(stable, probs) == pytest.approx(100)
    assert discrete_standard_deviation(risky, probs) == pytest.approx(2000)


def test_simulation_is_reproducible_and_close_to_model():
    a, b = simulate_profit(200_000, seed=4), simulate_profit(200_000, seed=4)
    assert np.array_equal(a, b)
    assert a.mean() == pytest.approx(expected_value(PROFIT_VALUES, PROFIT_PROBABILITIES), abs=12)
    assert a.var(ddof=0) == pytest.approx(discrete_variance(PROFIT_VALUES, PROFIT_PROBABILITIES), rel=.015)


def test_figures(tmp_path: Path):
    paths = create_figures(tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-18")
    assert chapter and chapter.available and chapter.title == "Expected Value & Variability"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Same mean, different risk" in output and "theoretical" in output
    assert len(output.splitlines()) < 25
