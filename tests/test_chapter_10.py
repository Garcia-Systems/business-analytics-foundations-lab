"""DataFrame transformations, summaries, figures, and execution for Chapter 10."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_foundations.chapter_10 import (
    category_totals, create_category_figure, create_daily_figure,
    create_location_figure, daily_revenue, high_value_downtown,
    location_metrics, prepare_transactions, run,
)
from analytics_foundations.chapters import get_chapter
from analytics_foundations.datasets import load_chapter_10_data

EXPECTED_COLUMNS = [
    "transaction_id", "date", "location", "category", "quantity",
    "unit_price", "discount", "customer_type",
]


def test_dataset_loads_at_transaction_grain() -> None:
    df = load_chapter_10_data()
    assert df.shape == (24, 8)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df["transaction_id"].is_unique
    assert df.notna().all().all()


def test_dates_and_derived_columns_are_correct() -> None:
    df = prepare_transactions()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "date"] == pd.Timestamp("2026-03-02")
    assert df.loc[0, "gross_revenue"] == pytest.approx(72)
    assert df.loc[2, "net_revenue"] == pytest.approx(36)
    assert df.loc[0, "day_of_week"] == "Monday"


def test_filtering_and_sorting() -> None:
    selected = high_value_downtown(prepare_transactions())
    assert not selected.empty
    assert selected["location"].eq("Downtown").all()
    assert selected["net_revenue"].gt(50).all()
    assert selected["net_revenue"].is_monotonic_decreasing
    assert selected.iloc[0]["transaction_id"] == "T024"


def test_grouped_metrics_reconcile_and_have_distinct_leaders() -> None:
    df = prepare_transactions()
    categories = category_totals(df)
    locations = location_metrics(df)
    assert set(categories["category"]) == {"Beverage", "Dessert", "Entree"}
    assert categories["revenue"].sum() == pytest.approx(df["net_revenue"].sum())
    assert locations["revenue"].sum() == pytest.approx(df["net_revenue"].sum())
    assert locations["transactions"].sum() == len(df)
    assert locations["units"].sum() == df["quantity"].sum()
    assert categories.iloc[0]["category"] == "Entree"
    assert locations.iloc[0].name == "Downtown"
    assert locations["average_transaction"].idxmax() == "Riverside"


def test_daily_grain_and_numpy_conversion_agree() -> None:
    df = prepare_transactions()
    daily = daily_revenue(df)
    assert daily["date"].is_unique
    assert daily["revenue"].sum() == pytest.approx(df["net_revenue"].sum())
    columns = ["quantity", "unit_price", "discount"]
    values = df[columns].to_numpy()
    assert isinstance(values, np.ndarray)
    assert values.shape == (24, 3)
    np.testing.assert_allclose(values, df[columns].values)


@pytest.mark.parametrize("creator,name", [
    (create_category_figure, "category.png"),
    (create_location_figure, "location.png"),
    (create_daily_figure, "daily.png"),
])
def test_figure_generation(tmp_path: Path, creator, name: str) -> None:
    path = tmp_path / name
    assert creator(prepare_transactions(), path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_chapter_registration_and_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chapter = get_chapter("chapter-10")
    assert chapter is not None and chapter.available and chapter.run is not None
    assert chapter.title == "Tables & DataFrames"
    assert run(tmp_path) == 0
    assert len(list(tmp_path.glob("chapter-10-*.png"))) == 3
    output = capsys.readouterr().out
    assert "Loaded shape: (24, 8)" in output
    assert "Highest-revenue category" in output
    assert "what one row represents" in output
