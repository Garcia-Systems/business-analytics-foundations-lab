"""Numerical, simulation, visual, and CLI tests for Chapter 23."""
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t
from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.chapter_23 import (REQUIRED_FIGURES, biased_interval_experiment,
    create_figures, mean_confidence_interval, required_sample_size,
    simulate_mean_interval_coverage, wilson_proportion_interval)
from analytics_foundations.chapters import get_chapter

@pytest.fixture
def population(): return generate_population().wait_minutes.to_numpy()

def test_mean_interval_anatomy_and_validation():
    values=np.array([10.,12.,14.,16.,18.]); result=mean_confidence_interval(values)
    assert result.estimate == 14 and result.sample_sd == pytest.approx(values.std(ddof=1))
    assert result.standard_error == pytest.approx(result.sample_sd/np.sqrt(5))
    assert result.critical_value == pytest.approx(t.ppf(.975,4)) and result.critical_value > 1.96
    assert result.lower < result.estimate < result.upper
    assert result.margin_of_error == pytest.approx((result.upper-result.lower)/2)
    for confidence in (0, 1, -1, 1.1):
        with pytest.raises(ValueError): mean_confidence_interval(values, confidence=confidence)
    with pytest.raises(ValueError): mean_confidence_interval([1])

def test_confidence_and_sample_size_change_precision():
    values=np.arange(1.,21.); assert mean_confidence_interval(values,confidence=.99).margin_of_error > mean_confidence_interval(values,confidence=.90).margin_of_error
    small=np.tile([0.,10.],5); large=np.tile([0.,10.],50)
    assert mean_confidence_interval(large).standard_error < mean_confidence_interval(small).standard_error
    assert required_sample_size(population_sd=8, margin_of_error=1) == 246

def test_coverage_structure_flags_and_empirical_result(population):
    frame=simulate_mean_interval_coverage(population,sample_size=64,confidence=.95,repetitions=5000,rng=np.random.default_rng(2301))
    assert frame.shape == (5000,5) and frame.covers_true_mean.dtype == bool
    truth=population.mean(); assert np.array_equal(frame.covers_true_mean,(frame.lower<=truth)&(truth<=frame.upper))
    assert frame.covers_true_mean.mean() == pytest.approx(.95, abs=.025)

def test_biased_interval_is_narrow_and_misses(population):
    result, truth=biased_interval_experiment(population)
    assert result.margin_of_error < 1 and not result.lower <= truth <= result.upper

def test_wilson_interval():
    low, high=wilson_proportion_interval(2,10); assert 0 < low < .2 < high < 1
    with pytest.raises(ValueError): wilson_proportion_interval(11,10)

def test_figures(population,tmp_path: Path):
    sample=np.random.default_rng(2).choice(population,64)
    coverage=simulate_mean_interval_coverage(population,sample_size=64,confidence=.95,repetitions=100,rng=np.random.default_rng(3))
    paths=create_figures(population,sample,coverage,tmp_path)
    assert [p.name for p in paths] == list(REQUIRED_FIGURES)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)

def test_registration_and_execution(tmp_path: Path,capsys):
    chapter=get_chapter("chapter-23"); assert chapter and chapter.available and chapter.title == "Estimation & Confidence"
    from analytics_foundations.chapter_23 import run
    assert run(tmp_path)==0; output=capsys.readouterr().out
    assert "95% t interval" in output and "no p-value" in output and len(output.splitlines()) < 20
