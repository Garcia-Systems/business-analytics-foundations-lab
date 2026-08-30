"""Linear systems, least squares, figures, and execution for Chapter 8."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_08 import (
    BUSINESS_A, BUSINESS_B, DEPENDENT_A, classify_system,
    create_least_squares_figure, create_system_figure, intercept_matrix,
    least_squares_fit, linear_combination, run, solve_exact_system,
)


def test_linear_combination_and_column_interpretation() -> None:
    vectors = np.array([[1, 3], [2, 1]])
    np.testing.assert_allclose(linear_combination(vectors, [2, 3]), [11, 7])
    np.testing.assert_allclose(vectors @ np.array([2, 3]), 2 * vectors[:, 0] + 3 * vectors[:, 1])


def test_exact_solution_and_verification() -> None:
    solution = solve_exact_system()
    np.testing.assert_allclose(solution, [32, 46])
    np.testing.assert_allclose(BUSINESS_A @ solution, BUSINESS_B)
    assert np.linalg.matrix_rank(BUSINESS_A) == 2
    assert classify_system(BUSINESS_A, BUSINESS_B) == "unique"


def test_dependent_and_inconsistent_systems() -> None:
    assert np.linalg.matrix_rank(DEPENDENT_A) == 1
    assert classify_system(DEPENDENT_A, [6, 12]) == "dependent"
    assert classify_system(DEPENDENT_A, [6, 15]) == "inconsistent"
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(DEPENDENT_A, [6, 15])


def test_intercept_and_least_squares_reconcile() -> None:
    x = np.arange(1., 6.)
    y = np.array([9., 12., 16., 19., 23.])
    X = intercept_matrix(x)
    np.testing.assert_allclose(X[:, 0], 1)
    np.testing.assert_allclose(X[:, 1], x)
    beta, predictions, residuals, rss, rank, singular_values = least_squares_fit(X, y)
    np.testing.assert_allclose(beta, [5.3, 3.5])
    np.testing.assert_allclose(predictions, X @ beta)
    np.testing.assert_allclose(residuals, y - predictions)
    assert residuals[0] == pytest.approx(.2)
    assert rss == pytest.approx(np.sum(residuals ** 2))
    assert rss == pytest.approx(.3)
    assert rank == 2
    assert singular_values.size == 2


@pytest.mark.parametrize("kind", ["unique", "dependent", "inconsistent"])
def test_system_figures(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"{kind}.png"
    assert create_system_figure(kind, path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_least_squares_figure(tmp_path: Path) -> None:
    x = np.arange(1., 6.)
    y = np.array([9., 12., 16., 19., 23.])
    beta = least_squares_fit(intercept_matrix(x), y)[0]
    path = tmp_path / "fit.png"
    assert create_least_squares_figure(x, y, beta, path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_chapter_experiment_executes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-08-*.png"))) == 4
    output = capsys.readouterr().out
    assert "residual sum of squares" in output
    assert "inconsistent; no exact solution" in output
    assert "Small residuals alone" in output
