"""Structural, numerical, visual, and CLI tests for Chapter 21."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_foundations.chapter_21 import (
    POPULATION_SIZE, REQUIRED_FIELDS, REQUIRED_FIGURES, biased_sample,
    composition_summary, create_figures, generate_population, population_parameters,
    random_sample, repeated_sample_means, sample_size_experiment, sample_statistics,
    stratified_sample,
)
from analytics_foundations.chapters import get_chapter


@pytest.fixture
def population():
    return generate_population()


def test_population_is_deterministic_clean_and_has_required_grain():
    first = generate_population(); second = generate_population()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == POPULATION_SIZE
    assert tuple(first.columns) == REQUIRED_FIELDS
    assert first.observation_id.is_unique and not first.isna().any().any()
    assert first.location_id.nunique() == 5 and (first.wait_minutes >= 0).all()


def test_random_sample_is_reproducible_valid_and_does_not_mutate(population):
    before = population.copy(deep=True)
    first = random_sample(population, n=40, seed=7)
    pd.testing.assert_frame_equal(first, random_sample(population, n=40, seed=7))
    assert len(first) == 40 and first.observation_id.is_unique
    assert set(first.observation_id) <= set(population.observation_id)
    pd.testing.assert_frame_equal(population, before)
    with pytest.raises(ValueError): random_sample(population, n=len(population) + 1)


def test_statistics_match_pandas_and_use_sample_sd(population):
    sample = random_sample(population, n=40); result = sample_statistics(sample)
    assert result["mean"] == pytest.approx(sample.wait_minutes.mean())
    assert result["standard_deviation"] == pytest.approx(sample.wait_minutes.std(ddof=1))
    assert result["proportion_over_20"] == pytest.approx(sample.wait_minutes.gt(20).mean())
    assert population_parameters(population)["standard_deviation"] == pytest.approx(population.wait_minutes.std(ddof=0))


def test_repeated_samples_and_size_experiment_have_expected_structure(population):
    repeated = repeated_sample_means(population)
    assert len(repeated) == 5 and repeated.mean_wait.nunique() > 1
    sizes = sample_size_experiment(population)
    assert sizes.groupby("n").size().to_dict() == {10: 5, 40: 5, 200: 5}
    assert np.isfinite(sizes.mean_wait).all()


def test_large_biased_sample_uses_restricted_subset_and_misses_downward(population):
    biased = biased_sample(population)
    assert len(biased) == 500 and set(biased.location_id) == {"Harbor"}
    assert biased.wait_minutes.mean() < population.wait_minutes.mean() - 2


def test_composition_and_stratification(population):
    random = random_sample(population); biased = biased_sample(population)
    composition = composition_summary(population, random, biased)
    assert np.allclose(composition[["population", "random_sample", "biased_sample"]].sum(), 1)
    assert composition.loc[composition.location_id == "Harbor", "biased_sample"].item() == 1
    stratified = stratified_sample(population, per_location=10)
    assert len(stratified) == 50 and (stratified.location_id.value_counts() == 10).all()


def test_figure_generation(population, tmp_path: Path):
    random = random_sample(population); biased = biased_sample(population)
    paths = create_figures(population, random, biased, repeated_sample_means(population),
                           sample_size_experiment(population), tmp_path)
    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-21")
    assert chapter and chapter.available and chapter.title == "Samples Tell Stories"
    from analytics_foundations.chapter_21 import run
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Target population:" in output and "Observational unit / grain:" in output
    assert "Large biased sample:" in output and "Next question—not answered here" in output
    assert len(output.splitlines()) < 35
