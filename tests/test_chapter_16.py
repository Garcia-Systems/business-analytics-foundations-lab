"""Random-variable calculations, simulation, figures, and CLI for Chapter 16."""

from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_16 import (
    PROBABILITIES, REQUIRED_FIGURES, VALUES, create_figures, discrete_cdf,
    empirical_proportions, exact_probability, indicator, probability_between,
    revenue_transform, run, simulate_discrete, threshold_probability, validate_pmf,
)
from analytics_foundations.chapters import get_chapter


def test_valid_pmf_is_accepted() -> None:
    assert validate_pmf(VALUES, PROBABILITIES) is None


@pytest.mark.parametrize(
    ("values", "probabilities", "message"),
    [
        ([1, 2], [.2, .2], "sum to 1"),
        ([1, 2], [1.1, -.1], "negative"),
        ([1], [.5, .5], "matching lengths"),
    ],
)
def test_invalid_pmf_is_rejected(values: list[int], probabilities: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_pmf(values, probabilities)


def test_exact_threshold_and_interval_probabilities() -> None:
    assert exact_probability(VALUES, PROBABILITIES, 160) == pytest.approx(.35)
    assert exact_probability(VALUES, PROBABILITIES, 999) == pytest.approx(0)
    assert threshold_probability(VALUES, PROBABILITIES, 200) == pytest.approx(.35)
    assert threshold_probability(VALUES, PROBABILITIES, 160, comparison="<") == pytest.approx(.30)
    assert probability_between(VALUES, PROBABILITIES, 120, 200) == pytest.approx(.80)


def test_cdf_is_correct_monotone_and_reaches_one() -> None:
    cdf = np.array([discrete_cdf(VALUES, PROBABILITIES, x) for x in VALUES])
    assert cdf == pytest.approx([.10, .30, .65, .90, 1.0])
    assert np.all(np.diff(cdf) >= 0)
    assert discrete_cdf(VALUES, PROBABILITIES, 1_000) == pytest.approx(1)
    assert discrete_cdf(VALUES, PROBABILITIES, 0) == pytest.approx(0)


def test_indicator_and_deterministic_transformations() -> None:
    assert indicator(VALUES).tolist() == [0, 0, 0, 1, 1]
    assert revenue_transform(VALUES).tolist() == [1440, 2160, 2880, 3600, 4320]
    assert np.maximum(VALUES - 180, 0).tolist() == [0, 0, 0, 20, 60]


def test_simulation_is_reproducible_and_stays_on_support() -> None:
    first = simulate_discrete(VALUES, PROBABILITIES, seed=160)
    second = simulate_discrete(VALUES, PROBABILITIES, seed=160)
    assert np.array_equal(first, second)
    assert set(first).issubset(set(VALUES))


def test_large_simulation_empirical_distribution_reflects_pmf() -> None:
    samples = simulate_discrete(VALUES, PROBABILITIES, size=100_000, seed=161)
    assert empirical_proportions(samples, VALUES) == pytest.approx(PROBABILITIES, abs=.006)


def test_all_figures_are_generated(tmp_path: Path) -> None:
    paths = create_figures(tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_chapter_registration_and_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chapter = get_chapter("chapter-16")
    assert chapter is not None and chapter.available
    assert chapter.title == "Random Variables"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "model PMF" in output
    assert "empirical proportions" in output
    assert len(output.splitlines()) < 20
