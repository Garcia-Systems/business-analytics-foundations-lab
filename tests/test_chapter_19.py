"""Numerical, visual, data, and CLI tests for Chapter 19."""
from pathlib import Path
import numpy as np
import pytest
from analytics_foundations.chapter_19 import (NUMERIC_FIELDS, REQUIRED_FIGURES, create_figures, load_data, run, sample_correlation, sample_covariance, variance_of_sum)
from analytics_foundations.chapters import get_chapter

def test_sample_covariance_formula_symmetry_and_variance_identity():
    x, y = np.array([1., 2., 3.]), np.array([2., 4., 5.])
    assert sample_covariance(x, y) == pytest.approx(1.5)
    assert sample_covariance(x, y) == pytest.approx(sample_covariance(y, x))
    assert sample_covariance(x, x) == pytest.approx(np.var(x, ddof=1))

def test_covariance_sign_and_scaling():
    x = np.arange(5.)
    assert sample_covariance(x, 2*x+10) > 0
    assert sample_covariance(x, 100-3*x) < 0
    assert sample_covariance(x, 100*(2*x+10)) == pytest.approx(100*sample_covariance(x, 2*x+10))

def test_correlation_properties_and_scaling():
    x, y = np.array([1., 2., 3.]), np.array([2., 4., 5.]); r = sample_correlation(x, y)
    assert r == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert r == pytest.approx(sample_correlation(y, x)) and -1 <= r <= 1
    assert sample_correlation(x, 100*y) == pytest.approx(r)
    assert sample_correlation(x, 2*x+10) == pytest.approx(1)
    assert sample_correlation(x, 100-3*x) == pytest.approx(-1)

@pytest.mark.parametrize("x,y,message", [([1], [2], "at least two"), ([1, 2], [3], "equal length"), ([1, 1], [2, 3], "zero variability"), ([1, np.inf], [2, 3], "finite")])
def test_validation_errors(x, y, message):
    with pytest.raises(ValueError, match=message): sample_correlation(x, y)

def test_nonlinear_zero_correlation_but_deterministic_dependence():
    x = np.arange(-3., 4.); y = x**2
    assert sample_correlation(x, y) == pytest.approx(0, abs=1e-15)
    assert np.array_equal(y, x**2)

def test_dataset_covariance_and_correlation_matrices():
    df = load_data(); cov, corr = df[NUMERIC_FIELDS].cov(), df[NUMERIC_FIELDS].corr()
    assert len(df) == 84 and not df.duplicated(["date", "location_id"]).any()
    assert cov.shape == (4, 4) and np.allclose(cov, cov.T)
    assert np.allclose(np.diag(corr), 1)

def test_variance_of_sum():
    assert variance_of_sum(100, 225, 50) == 425
    assert variance_of_sum(100, 225, 0) == 325
    assert variance_of_sum(100, 225, -50) == 225

def test_figures(tmp_path: Path):
    paths = create_figures(load_data(), tmp_path)
    assert [p.name for p in paths] == list(REQUIRED_FIGURES)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)

def test_registration_and_execution(tmp_path: Path, capsys):
    chapter = get_chapter("chapter-19")
    assert chapter and chapter.available and chapter.title == "Covariance & Dependence"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "does NOT" in output and "zero correlation is not independence" in output
    assert len(output.splitlines()) < 35
