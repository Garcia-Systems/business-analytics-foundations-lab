"""Calculations, simulations, figures, and registration for Chapter 17."""
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import binom, norm
from analytics_foundations.chapter_17 import (REQUIRED_FIGURES, bernoulli_probabilities,
    binomial_probability, binomial_tail, create_figures, interval_probability, run,
    simulate_distributions, uniform_model, z_score)
from analytics_foundations.chapters import get_chapter

def test_bernoulli_probabilities_and_support():
    assert bernoulli_probabilities(.2).sum() == pytest.approx(1)
    assert set(simulate_distributions(1000)["bernoulli"]) <= {0, 1}

def test_binomial_pmf_and_tail():
    assert binomial_probability(2, 5, .2) == pytest.approx(.2048)
    assert sum(binomial_probability(k, 20, .2) for k in range(21)) == pytest.approx(1)
    assert binomial_tail(2, 5, .2) == pytest.approx(1-binom.cdf(1, 5, .2))

def test_uniform_density_interval_and_parameterization():
    dist = uniform_model(2, 10)
    assert dist.kwds == {"loc": 2, "scale": 8}
    assert dist.pdf(5) == pytest.approx(1/8)
    assert dist.pdf(1) == 0 and dist.pdf(11) == 0
    assert interval_probability(uniform_model(0, 10), 2, 5) == pytest.approx(.3)

def test_normal_probabilities_and_standardization():
    dist = norm(12, 2)
    assert dist.cdf(15) == pytest.approx(.9331927987)
    assert interval_probability(dist, 10, 14) == pytest.approx(.6826894921)
    assert dist.sf(15) == pytest.approx(.0668072013)
    assert z_score(15, 12, 2) == pytest.approx(1.5)

def test_simulations_reproducible_supported_and_plausible():
    a, b = simulate_distributions(100_000, seed=9), simulate_distributions(100_000, seed=9)
    assert all(np.array_equal(a[key], b[key]) for key in a)
    assert np.all((a["binomial"] >= 0) & (a["binomial"] <= 20))
    empirical = np.bincount(a["binomial"], minlength=21) / len(a["binomial"])
    assert empirical == pytest.approx(binom.pmf(np.arange(21), 20, .2), abs=.004)
    assert a["normal"].mean() == pytest.approx(12, abs=.03)
    assert a["normal"].std() == pytest.approx(2, abs=.03)

def test_figures(tmp_path: Path):
    paths = create_figures(tmp_path)
    assert [p.name for p in paths] == list(REQUIRED_FIGURES)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)

def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-17")
    assert chapter and chapter.available and chapter.title == "Distributions"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Binomial assumptions" in output and "density is not point probability" in output.lower()
    assert len(output.splitlines()) < 20
