"""Grain, joins, reconciliation, figures, and CLI for Chapter 12."""
from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import MergeError

from analytics_foundations.chapter_12 import (
    GRAIN_LEDGER, aggregate_daily_revenue, aggregate_locations, attach_locations,
    build_daily_metrics, create_figures, demonstrate_labor_duplication,
    many_to_many_example, reconcile, run, validate_source_keys,
)
from analytics_foundations.chapters import get_chapter
from analytics_foundations.datasets import load_chapter_12_data


def sources():
    return load_chapter_12_data()


def test_table_grains_and_key_assumptions() -> None:
    transactions, locations, labor = sources()
    assert transactions.shape == (25, 6) and locations.shape == (3, 5) and labor.shape == (9, 4)
    assert transactions.transaction_id.is_unique
    assert locations.location_id.is_unique
    assert not transactions.location_id.is_unique
    assert not labor.location_id.is_unique and not labor.date.is_unique
    assert not labor.duplicated(["location_id", "date"]).any()
    validate_source_keys(transactions, locations, labor)
    assert GRAIN_LEDGER["daily_metrics"] == "location-day"


def test_source_validation_rejects_broken_grain() -> None:
    transactions, locations, labor = sources()
    with pytest.raises(ValueError, match="transaction_id"):
        validate_source_keys(pd.concat([transactions, transactions.iloc[[0]]]), locations, labor)
    with pytest.raises(ValueError, match="location_id"):
        validate_source_keys(transactions, pd.concat([locations, locations.iloc[[0]]]), labor)
    with pytest.raises(ValueError, match="location_id, date"):
        validate_source_keys(transactions, locations, pd.concat([labor, labor.iloc[[0]]]))


def test_many_to_one_metadata_join_and_unmatched_reporting() -> None:
    transactions, locations, _ = sources()
    joined, unmatched = attach_locations(transactions, locations)
    assert len(joined) == len(transactions) and unmatched == 0
    altered = transactions.copy(); altered.loc[0, "location_id"] = "UNKNOWN"
    joined, unmatched = attach_locations(altered, locations)
    assert len(joined) == len(altered) and unmatched == 1
    assert pd.isna(joined.loc[0, "location_name"])


def test_merge_validation_detects_wrong_cardinality() -> None:
    transactions, locations, _ = sources()
    duplicated = pd.concat([locations, locations.iloc[[0]]], ignore_index=True)
    with pytest.raises(MergeError):
        transactions.merge(duplicated, on="location_id", validate="many_to_one")


def test_aggregation_and_revenue_reconciliation() -> None:
    transactions, _, _ = sources(); daily = aggregate_daily_revenue(transactions)
    assert len(daily) == 9 and not daily.duplicated(["location_id", "date"]).any()
    assert daily.revenue.sum() == pytest.approx(transactions.net_revenue.sum())
    assert daily.transactions.sum() == len(transactions)


def test_daily_join_metrics_location_aggregation_and_reconciliation() -> None:
    transactions, locations, labor = sources()
    revenue = aggregate_daily_revenue(transactions); daily = build_daily_metrics(revenue, labor)
    assert len(daily) == 9 and daily.labor_hours.sum() == labor.labor_hours.sum()
    row = daily.iloc[0]
    assert row.revenue_per_labor_hour == pytest.approx(row.revenue / row.labor_hours)
    summary = aggregate_locations(daily, locations)
    assert len(summary) == 3 and summary.location_id.is_unique
    assert summary.revenue.sum() == pytest.approx(transactions.net_revenue.sum())
    assert summary.labor_hours.sum() == pytest.approx(labor.labor_hours.sum())
    assert all(reconcile(transactions, labor, daily, summary).values())


def test_ratio_of_sums_differs_from_average_of_ratios() -> None:
    example = pd.DataFrame({"revenue": [100, 900], "labor_hours": [10, 30]})
    assert (example.revenue / example.labor_hours).mean() == 20
    assert example.revenue.sum() / example.labor_hours.sum() == 25


def test_deliberate_labor_trap_inflates_but_does_not_mutate_source() -> None:
    transactions, _, labor = sources(); original = labor.copy(deep=True)
    repeated, naive = demonstrate_labor_duplication(transactions, labor)
    assert len(repeated) == len(transactions)
    assert naive > labor.labor_hours.sum()
    assert repeated.loc[(repeated.location_id == "L01") & (repeated.date == pd.Timestamp("2026-08-24")), "labor_hours"].tolist() == [20] * 4
    pd.testing.assert_frame_equal(labor, original)


def test_many_to_many_creates_four_combinations() -> None:
    result = many_to_many_example()
    assert len(result) == 4
    assert set(zip(result.order, result.campaign)) == {("A", "Email"), ("A", "SMS"), ("B", "Email"), ("B", "SMS")}


def test_figures_are_generated(tmp_path: Path) -> None:
    transactions, locations, labor = sources()
    daily = build_daily_metrics(aggregate_daily_revenue(transactions), labor)
    paths = create_figures(transactions, daily, aggregate_locations(daily, locations), tmp_path)
    assert len(paths) == 3
    assert all(path.exists() and path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_registration_and_cli_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chapter = get_chapter("chapter-12")
    assert chapter is not None and chapter.available and chapter.run is not None
    assert chapter.title == "Transform, Group & Join"
    assert run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "Deliberate bad join" in output and "Reconciliation passed" in output
    assert len(output.splitlines()) < 20
