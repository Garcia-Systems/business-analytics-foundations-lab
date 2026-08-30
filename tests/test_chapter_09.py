"""Array operations, visualizations, and execution for Chapter 9."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_09 import (
    CUSTOMERS, LABOR_HOURS, REVENUE, TARGETS, business_metrics,
    create_location_totals_figure, create_revenue_heatmap,
    create_target_deviation_figure, run, safe_slice, standardize, summarize,
    target_deviations,
)


def test_array_creation_shape_dimension_and_dtype() -> None:
    assert isinstance(REVENUE, np.ndarray)
    assert REVENUE.shape == CUSTOMERS.shape == LABOR_HOURS.shape == (5, 7)
    assert REVENUE.ndim == 2 and REVENUE.size == 35
    assert np.issubdtype(REVENUE.dtype, np.integer)
    assert np.issubdtype(LABOR_HOURS.dtype, np.floating)


def test_elementwise_arithmetic_metrics_and_slicing() -> None:
    np.testing.assert_array_equal(np.array([100, 120, 90]) * 2, [200, 240, 180])
    metrics = business_metrics()
    np.testing.assert_allclose(metrics["revenue_per_customer"][0, :2], [30, 4600 / 150])
    np.testing.assert_allclose(metrics["revenue_per_labor_hour"][0, :2], [4200 / 78, 57.5])
    np.testing.assert_array_equal(REVENUE[0, :3], [4200, 4600, 4400])
    assert REVENUE[:, 2].shape == (5,)
    assert REVENUE[1:3, 2:5].shape == (2, 3)


def test_boolean_mask_filters_both_conditions() -> None:
    mask = (REVENUE > 5000) & (LABOR_HOURS < 100)
    selected = REVENUE[mask]
    assert mask.shape == REVENUE.shape
    assert np.all(selected > 5000)
    assert np.all(LABOR_HOURS[mask] < 100)
    np.testing.assert_array_equal(
        selected,
        [5100, 6200, 5900, 5200, 6100, 5400, 5400, 6500, 6300, 5700,
         6400, 5200, 6000, 5800],
    )


def test_aggregations_axes_and_extreme_positions_reconcile() -> None:
    result = summarize()
    np.testing.assert_array_equal(result["by_location"], REVENUE.sum(axis=1))
    np.testing.assert_array_equal(result["by_day"], REVENUE.sum(axis=0))
    np.testing.assert_allclose(result["average_by_location"], REVENUE.mean(axis=1))
    assert result["total"] == result["by_location"].sum() == result["by_day"].sum()
    assert result["best_location"] == 2 and result["worst_location"] == 1
    assert result["best_day"] == 5 and result["worst_day"] == 0


def test_broadcasting_and_reshaping() -> None:
    assert TARGETS[:, np.newaxis].shape == (5, 1)
    deviations = target_deviations()
    assert deviations.shape == (5, 7)
    np.testing.assert_array_equal(deviations[0], REVENUE[0] - TARGETS[0])
    np.testing.assert_array_equal(deviations[:, 0], REVENUE[:, 0] - TARGETS)
    assert TARGETS.reshape(5, 1).shape == (5, 1)
    with pytest.raises(ValueError):
        target_deviations(REVENUE, TARGETS[:4])


def test_standardization_and_copy_safe_slice() -> None:
    z = standardize(np.array([10, 20, 30, 40]))
    assert z.mean() == pytest.approx(0)
    assert z.std(ddof=1) == pytest.approx(1)
    original = np.array([1, 2, 3, 4, 5])
    subset = safe_slice(original, 1, 4)
    subset[0] = 99
    np.testing.assert_array_equal(original, [1, 2, 3, 4, 5])


@pytest.mark.parametrize("creator,name", [
    (lambda path: create_revenue_heatmap(REVENUE, path), "heatmap.png"),
    (lambda path: create_location_totals_figure(REVENUE, path), "totals.png"),
    (lambda path: create_target_deviation_figure(REVENUE, TARGETS, path), "targets.png"),
])
def test_figure_generation(tmp_path: Path, creator, name: str) -> None:
    path = tmp_path / name
    assert creator(path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_chapter_experiment_executes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-09-*.png"))) == 3
    output = capsys.readouterr().out
    assert "Revenue shape=(5, 7)" in output
    assert "Broadcasting:" in output
    assert "totals do not establish efficiency" in output
