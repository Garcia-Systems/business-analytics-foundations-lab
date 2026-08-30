"""Matrix mathematics, data, figures, and execution tests for Chapter 7."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_07 import (
    FEATURE_COLUMNS,
    SCORE_WEIGHTS,
    create_table_matrix_figure,
    create_weighted_score_figure,
    dataframe_to_matrix,
    matrices_compatible,
    matrix_add,
    multiple_scores,
    run,
    scalar_multiply,
    weighted_scores,
)
from analytics_foundations.datasets import load_chapter_07_data


def test_dataframe_becomes_expected_matrix() -> None:
    df = load_chapter_07_data()
    matrix = dataframe_to_matrix(df)
    assert matrix.shape == (6, 3)
    assert matrix.ndim == 2
    np.testing.assert_allclose(matrix[:3], [[4, 25, 12], [7, 18, 6], [3, 40, 24]])
    assert matrix[1, 2] == 6
    np.testing.assert_allclose(matrix[:, 0], [4, 7, 3, 8, 5, 6])
    assert list(df.columns) == ["customer_id", *FEATURE_COLUMNS]


def test_transpose_addition_and_scalar_multiplication() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(a.T, [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_allclose(matrix_add(a, a), [[2, 4, 6], [8, 10, 12]])
    np.testing.assert_allclose(scalar_multiply(2, a), [[2, 4, 6], [8, 10, 12]])
    with pytest.raises(ValueError, match="equal"):
        matrix_add(a, np.ones((3, 2)))


def test_matrix_vector_scores_match_row_dot_products() -> None:
    matrix = dataframe_to_matrix(load_chapter_07_data())
    scores = weighted_scores(matrix)
    np.testing.assert_allclose(scores[:3], [16.5, 18.8, 22.0])
    for row, score in zip(matrix, scores, strict=True):
        assert score == pytest.approx(np.dot(row, SCORE_WEIGHTS))


def test_matrix_matrix_multiplication_and_dimensions() -> None:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_array_equal(a @ b, expected)
    np.testing.assert_array_equal(np.matmul(a, b), expected)
    assert matrices_compatible(a, b)
    assert not matrices_compatible(np.ones((6, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="inside dimensions"):
        multiple_scores(np.ones((6, 3)), np.ones((2, 2)))


def test_elementwise_and_matrix_products_are_distinct() -> None:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    np.testing.assert_array_equal(a * b, [[5, 12], [21, 32]])
    np.testing.assert_array_equal(a @ b, [[19, 22], [43, 50]])
    assert not np.array_equal(a * b, a @ b)


def test_two_weight_columns_produce_two_outputs() -> None:
    matrix = dataframe_to_matrix(load_chapter_07_data())
    weights = np.column_stack([SCORE_WEIGHTS, [1.5, 0.25, 0.1]])
    result = multiple_scores(matrix, weights)
    assert result.shape == (6, 2)
    np.testing.assert_allclose(result[:, 0], weighted_scores(matrix))
    np.testing.assert_allclose(result[0], [16.5, 13.45])


@pytest.mark.parametrize("kind", ["table", "score"])
def test_figure_generation(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "nested" / f"{kind}.png"
    df = load_chapter_07_data()
    if kind == "table":
        returned = create_table_matrix_figure(df, path)
    else:
        returned = create_weighted_score_figure(
            df.customer_id, weighted_scores(dataframe_to_matrix(df)), path
        )
    assert returned == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_experiment_executes_without_crashing_on_incompatibility(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-07-*.png"))) == 2
    output = capsys.readouterr().out
    assert "inside dimensions 3 and 2 do not match" in output
    assert "All scores X @ w" in output
    assert "not a validated" in output
