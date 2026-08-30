"""Numerical integration, visualization, and execution tests for Chapter 5."""
from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_05 import (
    arrival_antiderivative, arrival_rate, cash_flow_rate,
    create_accumulation_figure, create_refinement_figure, create_riemann_figure,
    create_signed_accumulation_figure, create_simpson_figure,
    create_trapezoid_figure, exact_arrivals, interval_width, left_sum,
    midpoint_sum, right_sum, run, simpson_rule, trapezoid_rule,
)


def test_interval_width() -> None:
    assert interval_width(0, 6, 6) == pytest.approx(1)
    assert interval_width(1, 3, 4) == pytest.approx(.5)
    with pytest.raises(ValueError): interval_width(0, 1, 0)


def test_hand_riemann_sums() -> None:
    assert left_sum(arrival_rate, 0, 6, 3) == pytest.approx(104)
    assert right_sum(arrival_rate, 0, 6, 3) == pytest.approx(104)
    assert midpoint_sum(arrival_rate, 0, 6, 3) == pytest.approx(110)


def test_trapezoid_and_simpson_rules() -> None:
    assert trapezoid_rule(arrival_rate, 0, 6, 3) == pytest.approx(104)
    assert trapezoid_rule(arrival_rate, 0, 6, 6) == pytest.approx(
        (left_sum(arrival_rate, 0, 6, 6) + right_sum(arrival_rate, 0, 6, 6)) / 2
    )
    assert simpson_rule(arrival_rate, 0, 6, 6) == pytest.approx(108)
    with pytest.raises(ValueError, match="even"): simpson_rule(arrival_rate, 0, 6, 3)
    with pytest.raises(ValueError): simpson_rule(arrival_rate, 0, 6, 0)


def test_exact_integral_and_array_behavior() -> None:
    np.testing.assert_allclose(arrival_rate(np.array([0, 3, 6])), [12, 21, 12])
    np.testing.assert_allclose(arrival_antiderivative(np.array([0, 6])), [0, 108])
    assert exact_arrivals() == pytest.approx(108)


def test_convergence() -> None:
    exact = exact_arrivals()
    for method in (left_sum, right_sum, midpoint_sum, trapezoid_rule):
        assert abs(method(arrival_rate, 0, 6, 48) - exact) < abs(method(arrival_rate, 0, 6, 3) - exact)


def test_signed_accumulation() -> None:
    assert simpson_rule(cash_flow_rate, 0, 4, 4) == pytest.approx(0)
    assert simpson_rule(lambda x: np.abs(cash_flow_rate(x)), 0, 4, 4) == pytest.approx(4)


@pytest.mark.parametrize("creator,name", [
    (create_accumulation_figure, "area.png"),
    (create_refinement_figure, "refine.png"),
    (create_trapezoid_figure, "trap.png"),
    (create_simpson_figure, "simpson.png"),
    (create_signed_accumulation_figure, "signed.png"),
])
def test_figure_generation(tmp_path: Path, creator, name: str) -> None:
    path = tmp_path / "nested" / name
    assert creator(path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_riemann_figure_generation_and_validation(tmp_path: Path) -> None:
    for method in ("left", "right", "midpoint"):
        assert create_riemann_figure(tmp_path / f"{method}.png", method).exists()
    with pytest.raises(ValueError): create_riemann_figure(tmp_path / "bad.png", "bad")


def test_experiment_executes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-05-*.png"))) == 8
    output = capsys.readouterr().out
    assert "customers/hour" in output
    assert "SciPy check after our implementations" in output
    assert "Numerical error and model error differ" in output
