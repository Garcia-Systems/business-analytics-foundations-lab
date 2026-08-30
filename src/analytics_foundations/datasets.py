"""Small, transparent helpers for loading the textbook's source data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_chapter_00_data() -> pd.DataFrame:
    """Load the clean daily observations for the Chapter 0 cafe."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-00-cafe-daily.csv"
    return pd.read_csv(path, parse_dates=["date"])


def load_chapter_03_data() -> pd.DataFrame:
    """Load the clean transaction observations for the Chapter 3 cafe."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-03-cafe-transactions.csv"
    return pd.read_csv(path)


def load_chapter_06_data() -> pd.DataFrame:
    """Load the fictional customer feature data used in Chapter 6."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-06-cafe-customers.csv"
    return pd.read_csv(path)


def load_chapter_07_data() -> pd.DataFrame:
    """Load the fictional customer matrix data used in Chapter 7."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-07-cafe-customers.csv"
    return pd.read_csv(path)


def load_chapter_10_data() -> pd.DataFrame:
    """Load clean transaction-grain restaurant data for Chapter 10."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-10-restaurant-transactions.csv"
    return pd.read_csv(path)


def load_chapter_11_data() -> pd.DataFrame:
    """Load raw, deliberately messy transaction data without parsing it."""

    path = PROJECT_ROOT / "data" / "raw" / "chapter-11-messy-transactions.csv"
    return pd.read_csv(path)


def load_chapter_12_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transaction-, location-, and location-day-grain Chapter 12 tables."""

    raw = PROJECT_ROOT / "data" / "raw"
    transactions = pd.read_csv(raw / "chapter-12-transactions.csv", parse_dates=["date"])
    locations = pd.read_csv(raw / "chapter-12-locations.csv")
    labor_daily = pd.read_csv(raw / "chapter-12-labor-daily.csv", parse_dates=["date"])
    return transactions, locations, labor_daily


def load_chapter_13_data() -> pd.DataFrame:
    """Load the analysis-ready location-day observations for Chapter 13."""

    path = PROJECT_ROOT / "data" / "processed" / "chapter-13-location-day.csv"
    return pd.read_csv(path, parse_dates=["date"], dtype={"promotion_active": "boolean"})
