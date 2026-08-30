"""Vector mathematics, dataset, visualization, and execution tests for Chapter 6."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_foundations.chapter_06 import (
    FEATURES, closest_pair, create_customer_points_figure, create_distance_figure,
    create_scale_figure, dot_product, euclidean_distance, feature_scale_example,
    magnitude, row_to_vector, run, scalar_multiply, vector_add, vector_subtract,
    weighted_revenue,
)
from analytics_foundations.datasets import load_chapter_06_data


def test_vector_arithmetic() -> None:
    a, b = np.array([2, 3]), np.array([4, 1])
    np.testing.assert_allclose(vector_add(a, b), [6, 4])
    np.testing.assert_allclose(vector_subtract(b, a), [2, -2])
    np.testing.assert_allclose(scalar_multiply(2, np.array([3, 4])), [6, 8])


def test_magnitude_and_distance_match_hand_calculations() -> None:
    assert magnitude([3, 4]) == pytest.approx(5)
    assert euclidean_distance([5, 30], [7, 25]) == pytest.approx(np.sqrt(29))
    assert magnitude(vector_subtract([7, 25], [5, 30])) == pytest.approx(np.sqrt(29))


def test_dot_products_and_weighted_combination() -> None:
    assert dot_product([2, 3], [4, 1]) == pytest.approx(11)
    assert weighted_revenue() == pytest.approx(106)
    assert weighted_revenue() == pytest.approx(np.array([10, 4, 2]) @ np.array([5, 8, 12]))


def test_invalid_vector_shapes_fail_clearly() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        vector_add([1, 2], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        magnitude([[3, 4]])


def test_dataset_is_small_clean_and_valid() -> None:
    df = load_chapter_06_data()
    assert list(df.columns) == ["customer_id", "visits", "average_order_value", "months_as_customer"]
    assert 8 <= len(df) <= 12
    assert df.customer_id.is_unique and not df.isna().any().any()
    assert (df[["visits", "average_order_value", "months_as_customer"]] > 0).all().all()


def test_dataframe_row_becomes_vector() -> None:
    df = load_chapter_06_data(); features = df[FEATURES]
    vector = row_to_vector(features.iloc[0])
    np.testing.assert_allclose(vector, [5, 30])
    assert vector.shape == (2,) and np.issubdtype(vector.dtype, np.floating)


def test_closest_pair_uses_dataframe_rows() -> None:
    df = load_chapter_06_data().set_index("customer_id")
    first, second, distance = closest_pair(df[FEATURES])
    assert {first, second} == {"A", "E"}
    assert distance == pytest.approx(np.sqrt(2))


def test_feature_scale_changes_nearest_observation() -> None:
    result = feature_scale_example()
    assert result["raw_frequent"] == pytest.approx(np.sqrt(149))
    assert result["raw_spend_alike"] == pytest.approx(np.sqrt(160001))
    assert result["scaled_frequent"] == pytest.approx(np.sqrt(49.0001))
    assert result["scaled_spend_alike"] == pytest.approx(np.sqrt(1.16))
    assert result["raw_nearest"] != result["scaled_nearest"]


@pytest.mark.parametrize("creator,name", [
    (lambda df, path: create_customer_points_figure(df, path), "points.png"),
    (lambda df, path: create_distance_figure(df, path), "distance.png"),
])
def test_customer_figure_generation(tmp_path: Path, creator, name: str) -> None:
    path = tmp_path / "nested" / name
    assert creator(load_chapter_06_data(), path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_scale_figure_generation(tmp_path: Path) -> None:
    path = tmp_path / "scale.png"
    assert create_scale_figure(path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_experiment_executes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-06-*.png"))) == 3
    output = capsys.readouterr().out
    assert "Closest raw-feature pair" in output
    assert "not clustering" in output
    assert "Weighted revenue" in output
