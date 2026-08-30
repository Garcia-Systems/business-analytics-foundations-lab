"""Calculation, aggregation, visualization, and execution tests for Chapter 3."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_foundations.chapter_03 import (
    add_transaction_revenue,
    arithmetic_mean,
    calculate_metrics,
    category_summary,
    create_contribution_figure,
    create_revenue_by_category_figure,
    run,
    weighted_average,
)
from analytics_foundations.datasets import load_chapter_03_data


@pytest.fixture
def transactions() -> pd.DataFrame:
    return add_transaction_revenue(load_chapter_03_data())


def test_transaction_revenue_calculation() -> None:
    source = pd.DataFrame({"quantity": [2, 3], "unit_price": [10.0, 4.0]})
    result = add_transaction_revenue(source)
    np.testing.assert_allclose(result["revenue"], [20.0, 12.0])
    assert "revenue" not in source


def test_overall_metrics(transactions: pd.DataFrame) -> None:
    metrics = calculate_metrics(transactions)
    assert metrics["total_revenue"] == pytest.approx(486.5)
    assert metrics["total_quantity"] == 77
    assert metrics["average_transaction_revenue"] == pytest.approx(486.5 / 15)
    assert metrics["simple_average_unit_price"] == pytest.approx(9.5333333333)
    assert metrics["weighted_average_unit_price"] == pytest.approx(486.5 / 77)


def test_mean_and_weighted_average_accept_numpy_arrays() -> None:
    values = np.array([12.0, 18.0, 15.0, 20.0])
    assert arithmetic_mean(values) == pytest.approx(16.25)
    assert weighted_average(np.array([20.0, 4.0]), np.array([1, 9])) == pytest.approx(5.6)
    with pytest.raises(ValueError, match="at least one"):
        arithmetic_mean(np.array([]))
    with pytest.raises(ValueError, match="same shape"):
        weighted_average([1, 2], [1])
    with pytest.raises(ValueError, match="must not be zero"):
        weighted_average([1, 2], [0, 0])


def test_grouped_revenue_and_contribution(transactions: pd.DataFrame) -> None:
    summary = category_summary(transactions)
    assert summary.loc["Food", "revenue"] == pytest.approx(229.0)
    assert summary.loc["Beverage", "revenue"] == pytest.approx(168.5)
    assert summary.loc["Dessert", "revenue"] == pytest.approx(89.0)
    assert summary["revenue"].sum() == pytest.approx(transactions["revenue"].sum())
    assert summary.loc["Food", "revenue_share_pct"] == pytest.approx(229 / 486.5 * 100)
    assert summary["revenue_share_pct"].sum() == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("creator", "filename"),
    [
        (create_revenue_by_category_figure, "revenue.png"),
        (create_contribution_figure, "shares.png"),
    ],
)
def test_figure_generation(
    transactions: pd.DataFrame, tmp_path: Path, creator, filename: str
) -> None:
    destination = tmp_path / "nested" / filename
    assert creator(category_summary(transactions), destination) == destination
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_experiment_generates_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(output_dir=tmp_path) == 0
    assert (tmp_path / "chapter-03-revenue-by-category.png").is_file()
    assert (tmp_path / "chapter-03-category-contributions.png").is_file()
    output = capsys.readouterr().out
    assert "loop total = NumPy total = pandas total" in output
    assert "Quantity-weighted price" in output
