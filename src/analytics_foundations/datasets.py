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
