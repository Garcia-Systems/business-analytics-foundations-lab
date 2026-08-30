"""Data grain, calculations, figures, and CLI for Chapter 13."""
from pathlib import Path

import pandas as pd
import pytest

from analytics_foundations.chapter_13 import (
    REQUIRED_FIGURES, create_figures, location_averages, location_totals,
    outlier_candidates, prepare_daily_series, revenue_summary, run, validate_grain,
)
from analytics_foundations.chapters import get_chapter
from analytics_foundations.datasets import load_chapter_13_data


def test_dataset_is_analysis_ready_at_location_day_grain() -> None:
    df=load_chapter_13_data()
    assert df.shape == (84, 12)
    assert df.location_id.nunique() == 4 and df.date.nunique() == 21
    assert not df.duplicated(["location_id","date"]).any()
    assert pd.api.types.is_datetime64_any_dtype(df.date)
    validate_grain(df)


def test_validation_rejects_duplicate_composite_key() -> None:
    df=load_chapter_13_data()
    with pytest.raises(ValueError,match="uniquely"):
        validate_grain(pd.concat([df,df.iloc[[0]]],ignore_index=True))


def test_summaries_and_grouped_revenue_reconcile() -> None:
    df=load_chapter_13_data(); summary=revenue_summary(df); totals=location_totals(df)
    assert summary["count"] == 84
    assert summary["mean"] == pytest.approx(df.revenue.mean())
    assert summary["median"] == pytest.approx(df.revenue.median())
    assert totals.revenue.sum() == pytest.approx(df.revenue.sum())
    assert totals.revenue.is_monotonic_increasing


def test_time_preparation_is_aggregated_and_ordered() -> None:
    df=load_chapter_13_data().sample(frac=1,random_state=13)
    daily=prepare_daily_series(df)
    assert len(daily)==21 and daily.date.is_monotonic_increasing
    assert daily.revenue.sum() == pytest.approx(df.revenue.sum())


def test_outlier_rows_and_aggregate_grain_are_retrievable() -> None:
    df=load_chapter_13_data(); unusual=outlier_candidates(df); averages=location_averages(df)
    assert len(averages)==4 and averages.location_name.is_unique
    assert not unusual.empty
    assert ((unusual.location_id=="L01") & (unusual.date==pd.Timestamp("2026-07-18"))).any()


def test_all_required_figures_are_generated(tmp_path: Path) -> None:
    paths=create_figures(load_chapter_13_data(),tmp_path)
    assert [p.name for p in paths] == list(REQUIRED_FIGURES)
    assert all(p.exists() and p.read_bytes().startswith(b"\x89PNG") for p in paths)


def test_registration_and_execution(tmp_path: Path,capsys: pytest.CaptureFixture[str]) -> None:
    chapter=get_chapter("chapter-13")
    assert chapter is not None and chapter.available and chapter.title=="Seeing Data"
    assert run(tmp_path)==0
    output=capsys.readouterr().out
    assert "grain = one location-day" in output
    assert "Unsupported conclusion" in output
    assert len(output.splitlines()) < 18
