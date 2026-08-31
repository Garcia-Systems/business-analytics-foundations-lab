"""Numerical, visual, and CLI tests for Chapter 22."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_22 import (
    REQUIRED_FIGURES, biased_sampling_experiment, create_figures,
    generate_skewed_population, sample_size_experiment,
    simulate_sample_means, simulate_sample_proportions,
    standard_error_mean, standard_error_proportion,
)
from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.chapters import get_chapter


@pytest.fixture
def population():
    return generate_population().wait_minutes.to_numpy()


def test_sample_means_shape_reproducibility_and_validation(population):
    first = simulate_sample_means(population, sample_size=40, n_repetitions=250,
                                  rng=np.random.default_rng(7))
    second = simulate_sample_means(population, sample_size=40, n_repetitions=250,
                                   rng=np.random.default_rng(7))
    assert first.shape == (250,) and np.array_equal(first, second)
    for kwargs in ({"sample_size": 0, "n_repetitions": 2},
                   {"sample_size": 2, "n_repetitions": 0}):
        with pytest.raises(ValueError):
            simulate_sample_means(population, rng=np.random.default_rng(1), **kwargs)
    with pytest.raises(ValueError):
        simulate_sample_means(np.array([]), sample_size=2, n_repetitions=2,
                              rng=np.random.default_rng(1))


def test_empirical_center_and_se_agree_with_theory(population):
    means = simulate_sample_means(population, sample_size=50, n_repetitions=20_000,
                                  rng=np.random.default_rng(8))
    theory = standard_error_mean(population.std(ddof=0), 50)
    assert means.mean() == pytest.approx(population.mean(), abs=.08)
    assert means.std(ddof=0) == pytest.approx(theory, rel=.03)


def test_standard_error_size_and_square_root_scaling():
    assert standard_error_mean(18, 81) == pytest.approx(2)
    assert standard_error_mean(12, 144) < standard_error_mean(12, 36)
    assert standard_error_mean(12, 144) == pytest.approx(standard_error_mean(12, 36) / 2)
    with pytest.raises(ValueError): standard_error_mean(-1, 10)


def test_proportion_theory_and_empirical_shrinkage():
    assert standard_error_proportion(.2, 100) == pytest.approx(.04)
    small = simulate_sample_proportions(.2, sample_size=25, n_repetitions=10_000,
                                        rng=np.random.default_rng(9))
    large = simulate_sample_proportions(.2, sample_size=400, n_repetitions=10_000,
                                        rng=np.random.default_rng(10))
    assert small.mean() == pytest.approx(.2, abs=.01)
    assert large.std(ddof=0) < small.std(ddof=0)
    with pytest.raises(ValueError): standard_error_proportion(1.1, 10)


def test_skewed_population_is_deterministic_and_clt_reduces_skewness():
    population = generate_skewed_population()
    assert np.array_equal(population, generate_skewed_population())
    rng = np.random.default_rng(11)
    one = simulate_sample_means(population, sample_size=1, n_repetitions=15_000, rng=rng)
    hundred = simulate_sample_means(population, sample_size=100, n_repetitions=15_000, rng=rng)
    skew = lambda x: np.mean((x-x.mean())**3) / np.mean((x-x.mean())**2)**1.5
    assert skew(one) > 2 and abs(skew(hundred)) < abs(skew(one)) / 3


def test_bias_experiment_reproducible_and_centered_away(population):
    first = biased_sampling_experiment(population)
    second = biased_sampling_experiment(population)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    unbiased, biased = first
    assert unbiased.mean() == pytest.approx(population.mean(), abs=.1)
    assert biased.mean() - population.mean() == pytest.approx(5, abs=.1)
    assert biased.std(ddof=0) == pytest.approx(unbiased.std(ddof=0))


def test_size_experiment_structure(population):
    result = sample_size_experiment(population, n_repetitions=5000)
    assert result.n.tolist() == [5, 20, 50, 200]
    assert np.allclose(result.empirical_se, result.theoretical_se, rtol=.05)
    assert result.theoretical_se.is_monotonic_decreasing


def test_figure_generation(population, tmp_path: Path):
    rng = np.random.default_rng(12); sample = rng.choice(population, 40)
    means = simulate_sample_means(population, sample_size=40, n_repetitions=1000, rng=rng)
    paths = create_figures(population, sample, means, tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-22")
    assert chapter and chapter.available and chapter.title == "Sampling Distributions"
    from analytics_foundations.chapter_22 import run
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "empirical SE=" in output and "n=30 is not a universal" in output
    assert "no interval is built here" in output and len(output.splitlines()) < 20
