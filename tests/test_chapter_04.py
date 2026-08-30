"""Model, numerical derivative, visualization, and execution tests for Chapter 4."""
from pathlib import Path
import numpy as np
import pytest
from analytics_foundations.chapter_04 import (
    average_rate_of_change, central_difference, cost,
    create_average_rate_figure, create_profit_marginal_figure,
    create_secant_tangent_figure, forward_difference, marginal_cost,
    marginal_profit, marginal_revenue, price, profit, revenue, run,
    zero_marginal_profit_quantity,
)

def test_business_functions() -> None:
    assert price(100) == pytest.approx(25)
    assert revenue(100) == pytest.approx(2500)
    assert cost(100) == pytest.approx(1200)
    assert profit(100) == pytest.approx(1300)
    np.testing.assert_allclose(revenue(np.array([0, 100])), [0, 2500])

def test_change_and_average_rate() -> None:
    assert profit(110) - profit(100) == pytest.approx(95)
    assert average_rate_of_change(profit, 100, 110) == pytest.approx(9.5)
    with pytest.raises(ValueError): average_rate_of_change(profit, 1, 1)

def test_analytical_and_marginal_derivatives() -> None:
    assert marginal_revenue(100) == pytest.approx(20)
    assert marginal_cost(100) == pytest.approx(10)
    assert marginal_profit(100) == pytest.approx(10)
    np.testing.assert_allclose(marginal_profit(np.array([100, 200, 250])), [10, 0, -5])

def test_numerical_derivatives_approach_analytical() -> None:
    assert forward_difference(profit, 100, .01) == pytest.approx(9.9995)
    assert central_difference(profit, 100, .01) == pytest.approx(10, abs=1e-9)
    assert abs(forward_difference(profit, 100, .01) - 10) < abs(forward_difference(profit, 100, 10) - 10)
    with pytest.raises(ValueError): forward_difference(profit, 100, 0)
    with pytest.raises(ValueError): central_difference(profit, 100, 0)

def test_zero_marginal_profit_matches_peak() -> None:
    q = zero_marginal_profit_quantity()
    assert q == pytest.approx(200)
    assert marginal_profit(q) == pytest.approx(0)
    assert profit(q) > profit(q - 1) and profit(q) > profit(q + 1)

@pytest.mark.parametrize("creator,name", [
    (create_average_rate_figure, "average.png"),
    (create_secant_tangent_figure, "tangent.png"),
    (create_profit_marginal_figure, "marginal.png"),
])
def test_figure_generation(tmp_path: Path, creator, name: str) -> None:
    path = tmp_path / "nested" / name
    assert creator(path) == path
    assert path.read_bytes().startswith(b"\x89PNG")

def test_experiment_executes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-04-*.png"))) == 3
    output = capsys.readouterr().out
    assert "Shrinking h" in output
    assert "dollars of profit per unit" in output
    assert "total profit there is $1,800.00, not $0" in output
