"""Probability helpers, figures, simulation, and CLI for Chapter 14."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_14 import (
    BUSY_OR_HIGHER, DEMAND_MODEL, REQUIRED_FIGURES, complement_probability,
    create_figures, empirical_probability, intersection_probability,
    probability_of_event, run, simulate_categorical_outcomes,
    union_probability, validate_probability_model,
)
from analytics_foundations.chapters import get_chapter


def test_valid_probability_model_is_accepted() -> None:
    assert validate_probability_model(DEMAND_MODEL) is None


@pytest.mark.parametrize("model", [
    {"A": .2, "B": .7}, {"A": -.1, "B": 1.1}, {"A": 1.2, "B": -.2},
])
def test_invalid_probability_models_are_rejected(model: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        validate_probability_model(model)


def test_event_complement_and_mutually_exclusive_union() -> None:
    assert probability_of_event(DEMAND_MODEL, {"Busy"}) == pytest.approx(.30)
    assert probability_of_event(DEMAND_MODEL, BUSY_OR_HIGHER) == pytest.approx(.50)
    assert complement_probability(DEMAND_MODEL, BUSY_OR_HIGHER) == pytest.approx(.50)
    assert union_probability(DEMAND_MODEL, {"Low"}, {"Very Busy"}) == pytest.approx(.35)
    with pytest.raises(ValueError, match="outside the sample space"):
        probability_of_event(DEMAND_MODEL, {"Impossible category"})


def test_overlapping_union_and_intersection() -> None:
    die = {str(i): 1 / 6 for i in range(1, 7)}
    even, high = {"2", "4", "6"}, {"4", "5", "6"}
    assert intersection_probability(die, even, high) == pytest.approx(2 / 6)
    assert union_probability(die, even, high) == pytest.approx(4 / 6)
    assert union_probability(die, even, high) == pytest.approx(
        probability_of_event(die, even) + probability_of_event(die, high)
        - intersection_probability(die, even, high))


def test_empirical_probability_and_reproducible_simulation() -> None:
    first = simulate_categorical_outcomes(DEMAND_MODEL, 100, seed=14)
    second = simulate_categorical_outcomes(DEMAND_MODEL, 100, seed=14)
    assert np.array_equal(first, second)
    assert set(first) <= set(DEMAND_MODEL)
    assert 0 <= empirical_probability(first, BUSY_OR_HIGHER) <= 1


def test_large_simulation_reasonably_approximates_model() -> None:
    outcomes = simulate_categorical_outcomes(DEMAND_MODEL, 100_000, seed=140)
    assert empirical_probability(outcomes, BUSY_OR_HIGHER) == pytest.approx(.5, abs=.01)


def test_all_figures_are_generated(tmp_path: Path) -> None:
    paths = create_figures(tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_chapter_registration_and_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chapter = get_chapter("chapter-14")
    assert chapter is not None and chapter.available and chapter.title == "Events & Probability"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Sample space" in output and "Not guaranteed" in output
    assert len(output.splitlines()) < 22
