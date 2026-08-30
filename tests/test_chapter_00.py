"""Behavior and calculation tests for the Chapter 0 experiment."""

from pathlib import Path

import pandas as pd
import pytest

from analytics_foundations.chapter_00 import (
    calculate_metrics,
    create_revenue_figure,
    run,
)
from analytics_foundations.datasets import load_chapter_00_data


EXPECTED_COLUMNS = {"date", "day_of_week", "customers", "revenue", "labor_hours"}


def test_dataset_loads_with_expected_clean_columns() -> None:
    data = load_chapter_00_data()
    assert len(data) == 14
    assert set(data.columns) == EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(data["date"])
    assert not data.isna().any().any()
    assert (data[["customers", "revenue", "labor_hours"]] > 0).all().all()


def test_core_metrics_have_expected_values() -> None:
    metrics = calculate_metrics(load_chapter_00_data())
    assert metrics["total_revenue"] == pytest.approx(11882.3)
    assert metrics["average_daily_revenue"] == pytest.approx(848.735714)
    assert metrics["total_customers"] == 970
    assert metrics["average_revenue_per_customer"] == pytest.approx(12.2497938)
    assert metrics["revenue_per_labor_hour"] == pytest.approx(40.5539249)


def test_figure_generation_writes_nonempty_png(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "revenue.png"
    result = create_revenue_figure(load_chapter_00_data(), destination)
    assert result == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_experiment_writes_expected_figure(tmp_path: Path) -> None:
    assert run(output_dir=tmp_path) == 0
    assert (tmp_path / "chapter-00-revenue-by-date.png").is_file()
