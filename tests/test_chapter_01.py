"""Calculation, visualization, and execution tests for Chapter 1."""

from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_01 import (
    break_even_quantity,
    cost,
    create_profit_figure,
    create_revenue_cost_figure,
    operational_break_even_quantity,
    profit,
    revenue,
    run,
)


def test_scalar_model_calculations() -> None:
    assert revenue(40) == 480
    assert cost(40) == 500
    assert profit(40) == -20
    assert revenue(60) == 720
    assert cost(60) == 600
    assert profit(60) == 120


def test_break_even_calculations() -> None:
    mathematical = break_even_quantity()
    assert mathematical == pytest.approx(300 / 7)
    assert revenue(mathematical) == pytest.approx(cost(mathematical))
    assert operational_break_even_quantity() == 43
    assert profit(42) < 0 < profit(43)


def test_break_even_requires_positive_contribution_margin() -> None:
    with pytest.raises(ValueError, match="price must be greater"):
        break_even_quantity(price=5, variable_cost=5)


def test_models_accept_numpy_arrays() -> None:
    quantities = np.array([0, 10, 20, 30])
    np.testing.assert_array_equal(revenue(quantities), [0, 120, 240, 360])
    np.testing.assert_array_equal(cost(quantities), [300, 350, 400, 450])
    np.testing.assert_array_equal(profit(quantities), [-300, -230, -160, -90])


@pytest.mark.parametrize(
    ("creator", "name"),
    [
        (create_revenue_cost_figure, "revenue-cost.png"),
        (create_profit_figure, "profit.png"),
    ],
)
def test_figure_generation_writes_png(tmp_path: Path, creator, name: str) -> None:
    destination = tmp_path / "nested" / name
    assert creator(np.arange(0, 101), destination) == destination
    assert destination.stat().st_size > 0
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_experiment_generates_both_figures(tmp_path: Path) -> None:
    assert run(output_dir=tmp_path) == 0
    assert (tmp_path / "chapter-01-revenue-and-cost.png").is_file()
    assert (tmp_path / "chapter-01-profit.png").is_file()
