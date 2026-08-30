"""Quality detection, documented cleaning, artifacts, and CLI for Chapter 11."""
from pathlib import Path
import hashlib

import pandas as pd
import pytest

from analytics_foundations.chapter_11 import (
    add_quality_flags, clean_transactions, create_figures,
    identify_conflicting_ids, normalize_categories, numeric_columns,
    outlier_candidates, parse_dates, quality_audit, run, validate_cleaned,
)
from analytics_foundations.chapters import get_chapter
from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_11_data

RAW_PATH = PROJECT_ROOT / "data/raw/chapter-11-messy-transactions.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_dataset_contains_intended_quality_problems() -> None:
    df = load_chapter_11_data()
    assert df.shape == (42, 10)
    assert not pd.api.types.is_numeric_dtype(df["quantity"]) and "five" in set(df["quantity"])
    assert df[["quantity", "unit_price", "category", "customer_type"]].isna().any().all()
    assert df.duplicated().sum() == 1
    assert identify_conflicting_ids(df) == {"T032"}
    assert {"Downtown", "downtown", "Downtown ", "DOWNTOWN"}.issubset(set(df["location"]))
    assert {"Beverage", "beverage", "Drinks"}.issubset(set(df["category"].dropna()))


def test_audit_reports_missing_duplicates_and_numeric_ranges() -> None:
    issues, structure = quality_audit(load_chapter_11_data())
    counts = issues.set_index("issue")["count"]
    assert counts["missing quantity"] == 1
    assert counts["exact duplicate rows"] == 1
    assert counts["duplicate ID rows"] == 4
    assert counts["conflicting IDs"] == 1
    assert structure.loc["quantity", "missing_percent"] == pytest.approx(100 / 42)
    assert structure.loc["discount", "maximum"] == 1.5
    assert structure.loc["quantity", "minimum"] == -2


def test_category_normalization_and_approved_semantic_mapping() -> None:
    result = normalize_categories(pd.Series([" Beverage ", "beverage", "Drinks", "Entree", None]))
    assert result.iloc[:3].tolist() == ["Beverage"] * 3
    assert result.iloc[3] == "Entree" and pd.isna(result.iloc[4])


def test_date_parsing_distinguishes_mixed_from_invalid() -> None:
    result = parse_dates(pd.Series(["2026-09-01", "09/02/2026", "2026/09/03", "Sep 4 2026", "2026-13-05"]))
    assert result.notna().sum() == 4
    assert result.iloc[1] == pd.Timestamp("2026-09-02")
    assert pd.isna(result.iloc[-1])


def test_numeric_conversion_and_business_rule_flags() -> None:
    raw = load_chapter_11_data()
    converted = numeric_columns(raw)
    assert pd.api.types.is_numeric_dtype(converted["quantity"])
    assert pd.isna(converted.loc[raw["quantity"].eq("five"), "quantity"]).all()
    flags = add_quality_flags(raw).set_index("transaction_id")
    assert flags.loc["T011", "invalid_quantity"]
    assert flags.loc["T012", "invalid_unit_price"]
    assert flags.loc["T013", "invalid_discount"]
    assert flags.loc["T015", "invalid_date"]
    assert flags.loc["T016", "invalid_labor_hours"]


def test_iqr_candidates_are_flags_not_deletions() -> None:
    values = pd.Series([10, 11, 12, 13, 1250])
    flags = outlier_candidates(values)
    assert flags.tolist() == [False, False, False, False, True]


def test_cleaning_decisions_validation_and_outlier_retention() -> None:
    cleaned, audit, log = clean_transactions(load_chapter_11_data())
    validate_cleaned(cleaned)
    assert len(cleaned) == 31
    assert "T020" in set(cleaned["transaction_id"])  # one exact copy remains
    assert "T032" not in set(cleaned["transaction_id"])  # both conflicting claims excluded
    assert cleaned.set_index("transaction_id").loc["T008", "unit_price"] == 20
    assert cleaned.set_index("transaction_id").loc["T014", "customer_type"] == "Unknown"
    assert cleaned.set_index("transaction_id").loc["T027", "outlier_candidate"]
    assert ((log.row_identifier == "T027") & (log.action == "RETAIN")).any()
    assert ((log.row_identifier == "T032") & (log.action == "EXCLUDE")).any()
    assert set(log.action).issuperset({"CORRECT", "STANDARDIZE", "IMPUTE", "EXCLUDE", "RETAIN"})
    assert not audit.empty


def test_pipeline_is_deterministic_and_does_not_mutate_input() -> None:
    raw = load_chapter_11_data(); original = raw.copy(deep=True)
    first, _, first_log = clean_transactions(raw)
    second, _, second_log = clean_transactions(raw)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_log, second_log)


def test_processed_generation_preserves_raw_and_figures(tmp_path: Path) -> None:
    before = digest(RAW_PATH); processed = tmp_path / "processed/clean.csv"; figures = tmp_path / "figures"
    assert run(figures, processed) == 0
    assert digest(RAW_PATH) == before
    written = pd.read_csv(processed)
    assert len(written) == 31 and written["transaction_id"].is_unique
    written_log = pd.read_csv(processed.with_name("chapter-11-cleaning-log.csv"))
    assert set(written_log["action"]).issuperset({"CORRECT", "EXCLUDE", "RETAIN"})
    paths = sorted(figures.glob("chapter-11-*.png"))
    assert len(paths) == 4 and all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_direct_figure_generation(tmp_path: Path) -> None:
    raw = load_chapter_11_data(); cleaned, _, _ = clean_transactions(raw)
    paths = create_figures(raw, cleaned, tmp_path)
    assert len(paths) == 4 and all(path.exists() for path in paths)


def test_registration_and_execution_is_concise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chapter = get_chapter("chapter-11")
    assert chapter is not None and chapter.available and chapter.run is not None
    assert chapter.title == "Messy Data"
    assert run(tmp_path / "figures", tmp_path / "clean.csv") == 0
    output = capsys.readouterr().out
    assert "inspect → detect → diagnose" in output
    assert "conflicting IDs=['T032']" in output
    assert "Post-clean validation passed" in output
    assert "Unresolved limitation" in output
