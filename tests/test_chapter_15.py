"""Conditional probability, Bayes, simulation, figures, and CLI for Chapter 15."""

from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_15 import (
    REQUIRED_FIGURES, are_independent, bayes_binary, build_day_dataset,
    conditional_probability, contingency_table, create_figures,
    fraud_count_table, run, simulate_fraud_alerts,
)
from analytics_foundations.chapters import get_chapter


def test_day_dataset_grain_and_contingency_counts() -> None:
    df = build_day_dataset()
    assert len(df) == 100
    assert df["date"].is_unique
    assert list(df.columns) == [
        "date", "is_friday", "busy", "promotion_active", "rain",
    ]
    assert contingency_table(df).to_dict() == {
        "Busy": {"Friday": 18, "Not Friday": 22, "Total": 40},
        "Not Busy": {"Friday": 6, "Not Friday": 54, "Total": 60},
        "Total": {"Friday": 24, "Not Friday": 76, "Total": 100},
    }


def test_marginal_joint_conditional_and_reverse_probabilities() -> None:
    table = contingency_table(build_day_dataset())
    p_busy = table.loc["Total", "Busy"] / table.loc["Total", "Total"]
    p_friday = table.loc["Friday", "Total"] / table.loc["Total", "Total"]
    p_joint = table.loc["Friday", "Busy"] / table.loc["Total", "Total"]
    forward = conditional_probability(p_joint, p_friday)
    reverse = conditional_probability(p_joint, p_busy)
    assert p_busy == pytest.approx(.40)
    assert p_friday == pytest.approx(.24)
    assert p_joint == pytest.approx(.18)
    assert forward == pytest.approx(.75)
    assert reverse == pytest.approx(.45)
    assert forward != reverse
    assert forward * p_friday == pytest.approx(p_joint)
    assert reverse * p_busy == pytest.approx(p_joint)


def test_independence_and_mutual_exclusivity_distinction() -> None:
    assert are_independent(.4, .5, .2)
    assert not are_independent(.4, .5, .1)
    assert not are_independent(.15, .20, 0)  # nonzero mutually exclusive events
    assert not are_independent(.40, .24, .18)  # Busy and Friday


def test_invalid_conditioning_and_probability_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        conditional_probability(0, 0)
    with pytest.raises(ValueError, match="cannot exceed"):
        conditional_probability(.3, .2)
    with pytest.raises(ValueError, match="zero probability"):
        bayes_binary(.5, 0, 0)


def test_bayes_formula_reconciles_with_count_table() -> None:
    counts = fraud_count_table()
    assert (counts.fraud, counts.no_fraud) == (100, 9_900)
    assert (counts.true_alerts, counts.false_alerts) == (90, 495)
    assert counts.total_alerts == 585
    expected = 90 / 585
    assert counts.posterior == pytest.approx(expected)
    assert bayes_binary(.01, .90, .05) == pytest.approx(expected)


def test_simulation_is_reproducible_and_approximates_bayes() -> None:
    first_fraud, first_alert = simulate_fraud_alerts(seed=150)
    second_fraud, second_alert = simulate_fraud_alerts(seed=150)
    assert np.array_equal(first_fraud, second_fraud)
    assert np.array_equal(first_alert, second_alert)
    assert first_fraud[first_alert].mean() == pytest.approx(
        bayes_binary(.01, .90, .05), abs=.015,
    )


def test_all_figures_are_generated(tmp_path: Path) -> None:
    paths = create_figures(tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_chapter_registration_and_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    chapter = get_chapter("chapter-15")
    assert chapter is not None and chapter.available
    assert chapter.title == "Conditional Probability"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Conditioning changes the denominator" in output
    assert "association is not causation" in output
    assert len(output.splitlines()) < 25
