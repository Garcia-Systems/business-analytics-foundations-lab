"""Calculation, visualization, and execution tests for Chapter 2."""

from pathlib import Path

import numpy as np
import pytest

from analytics_foundations.chapter_02 import (
    create_growth_figure,
    create_repeated_multiplication_figure,
    crossover_month,
    doubling_time,
    exponential_customers,
    linear_customers,
    percentage_change,
    percentage_point_change,
    run,
)


def test_linear_growth_calculations() -> None:
    assert linear_customers(0) == 500
    assert linear_customers(3) == 650
    assert linear_customers(12) == 1100


def test_exponential_growth_and_exponent_meanings() -> None:
    assert exponential_customers(0) == pytest.approx(500)
    assert exponential_customers(3) == pytest.approx(629.856)
    assert 2**0 == 1
    assert 2**-1 == pytest.approx(0.5)
    assert 2**0.5 == pytest.approx(np.sqrt(2))


def test_models_accept_numpy_arrays() -> None:
    months = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(linear_customers(months), [500, 550, 600, 650])
    np.testing.assert_allclose(exponential_customers(months), [500, 540, 583.2, 629.856])


def test_percentage_and_percentage_point_changes() -> None:
    assert percentage_change(40_000, 46_000) == pytest.approx(15)
    assert percentage_change(0.20, 0.25) == pytest.approx(25)
    assert percentage_point_change(0.20, 0.25) == pytest.approx(5)
    np.testing.assert_allclose(percentage_change([100, 200], [110, 240]), [10, 20])
    with pytest.raises(ValueError, match="old value is zero"):
        percentage_change(0, 10)


def test_doubling_time() -> None:
    assert doubling_time(0.08) == pytest.approx(9.006468342)
    with pytest.raises(ValueError, match="positive"):
        doubling_time(0)


def test_crossover_month() -> None:
    assert crossover_month() == 7
    assert exponential_customers(6) < linear_customers(6)
    assert exponential_customers(7) > linear_customers(7)
    with pytest.raises(ValueError, match="no crossover"):
        crossover_month(6)


@pytest.mark.parametrize(
    ("creator", "name", "months"),
    [
        (create_growth_figure, "growth.png", np.arange(25)),
        (create_repeated_multiplication_figure, "multiplication.png", np.arange(13)),
    ],
)
def test_figure_generation(tmp_path: Path, creator, name: str, months) -> None:
    destination = tmp_path / "nested" / name
    assert creator(months, destination) == destination
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_experiment_generates_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(output_dir=tmp_path) == 0
    assert (tmp_path / "chapter-02-linear-vs-exponential.png").is_file()
    assert (tmp_path / "chapter-02-repeated-multiplication.png").is_file()
    output = capsys.readouterr().out
    assert "Doubling time" in output
    assert "Mathematical consistency" in output
