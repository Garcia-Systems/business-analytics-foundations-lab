"""Auditable, business-rule-driven cleaning for Chapter 11."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics_foundations.datasets import PROJECT_ROOT, load_chapter_11_data

EXPECTED_CATEGORIES = {"Beverage", "Catering", "Dessert", "Entree"}
CATEGORY_MAP = {"beverage": "Beverage", "drinks": "Beverage", "dessert": "Dessert", "entree": "Entree", "catering": "Catering"}


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed, recognizable representations; invalid dates become NaT."""
    return pd.to_datetime(values, format="mixed", errors="coerce")


def normalize_categories(values: pd.Series) -> pd.Series:
    """Normalize formatting and apply the approved Drinks -> Beverage mapping."""
    keys = values.astype("string").str.strip().str.lower()
    return keys.map(CATEGORY_MAP).astype("string")


def numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric evidence, coercing dirty tokens so they remain detectable."""
    result = df.copy()
    for column in ["quantity", "unit_price", "discount", "labor_hours", "gross_revenue"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def identify_conflicting_ids(df: pd.DataFrame) -> set[str]:
    """Return repeated identifiers whose rows are not exact duplicates."""
    no_exact_repeats = df.drop_duplicates()
    counts = no_exact_repeats["transaction_id"].value_counts()
    return set(counts[counts > 1].index)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Expose rather than hide date, identifier, and numeric rule failures."""
    flagged = numeric_columns(df)
    flagged["parsed_date"] = parse_dates(flagged["date"])
    flagged["category_clean"] = normalize_categories(flagged["category"])
    flagged["invalid_date"] = flagged["parsed_date"].isna()
    flagged["invalid_quantity"] = flagged["quantity"].isna() | flagged["quantity"].le(0)
    flagged["invalid_unit_price"] = flagged["unit_price"].notna() & flagged["unit_price"].le(0)
    flagged["invalid_discount"] = flagged["discount"].isna() | ~flagged["discount"].between(0, 1)
    flagged["invalid_labor_hours"] = flagged["labor_hours"].isna() | flagged["labor_hours"].lt(0)
    flagged["duplicate_id"] = flagged["transaction_id"].duplicated(keep=False)
    return flagged


def outlier_candidates(values: pd.Series) -> pd.Series:
    """Flag IQR candidates; this is an investigation prompt, not an error label."""
    valid = values.dropna()
    q1, q3 = valid.quantile([.25, .75])
    iqr = q3 - q1
    return values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)


def quality_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact issue counts and a per-column structural summary."""
    flagged = add_quality_flags(df)
    missing = df.isna().sum()
    numeric = numeric_columns(df)
    structure = pd.DataFrame({"missing_count": missing, "missing_percent": df.isna().mean().mul(100), "unique_count": df.nunique(dropna=True)})
    structure["minimum"] = pd.Series({column: numeric[column].min() for column in ["quantity", "unit_price", "discount", "labor_hours", "gross_revenue"]})
    structure["maximum"] = pd.Series({column: numeric[column].max() for column in ["quantity", "unit_price", "discount", "labor_hours", "gross_revenue"]})
    issues = pd.DataFrame({"issue": ["rows", "columns", "missing quantity", "missing unit_price", "missing category", "missing customer_type", "exact duplicate rows", "duplicate ID rows", "conflicting IDs", "invalid dates", "impossible quantities", "invalid unit prices", "invalid discounts", "invalid labor hours"],
        "count": [len(df), len(df.columns), int(missing["quantity"]), int(missing["unit_price"]), int(missing["category"]), int(missing["customer_type"]), int(df.duplicated().sum()), int(df["transaction_id"].duplicated(keep=False).sum()), len(identify_conflicting_ids(df)), int(flagged["invalid_date"].sum()), int(flagged["invalid_quantity"].sum()), int(flagged["invalid_unit_price"].sum()), int(flagged["invalid_discount"].sum()), int(flagged["invalid_labor_hours"].sum())]})
    return issues, structure


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply explicit decisions and return cleaned data, audit, and audit trail."""
    audit, _ = quality_audit(raw)
    work = raw.copy()
    log: list[dict[str, object]] = []
    exact_mask = work.duplicated()
    for row in work.loc[exact_mask].itertuples():
        log.append({"row_identifier": row.transaction_id, "issue": "exact duplicate", "column": "all", "action": "EXCLUDE", "reason": "identical repeated observation"})
    work = work.loc[~exact_mask].copy()
    conflicts = identify_conflicting_ids(work)
    for identifier in sorted(conflicts):
        log.append({"row_identifier": identifier, "issue": "conflicting identifier", "column": "transaction_id", "action": "EXCLUDE", "reason": "requires source-system investigation"})
    work = work.loc[~work["transaction_id"].isin(conflicts)].copy()
    work = add_quality_flags(work)

    reconstruct = work["unit_price"].isna() & work["quantity"].gt(0) & work["gross_revenue"].notna()
    work.loc[reconstruct, "unit_price"] = work.loc[reconstruct, "gross_revenue"] / work.loc[reconstruct, "quantity"]
    for identifier in work.loc[reconstruct, "transaction_id"]:
        log.append({"row_identifier": identifier, "issue": "missing reconstructable value", "column": "unit_price", "action": "CORRECT", "reason": "gross_revenue / quantity under stated business rule"})
    missing_customer = work["customer_type"].isna()
    work.loc[missing_customer, "customer_type"] = "Unknown"
    for identifier in work.loc[missing_customer, "transaction_id"]:
        log.append({"row_identifier": identifier, "issue": "missing noncritical category", "column": "customer_type", "action": "IMPUTE", "reason": "preserve observation without inventing membership"})
    standardized = work["category"].notna() & work["category"].astype("string").str.strip().ne(work["category_clean"])
    for identifier in work.loc[standardized, "transaction_id"]:
        log.append({"row_identifier": identifier, "issue": "category variant", "column": "category", "action": "STANDARDIZE", "reason": "format normalization or approved Drinks mapping"})
    work["location_clean"] = work["location"].astype("string").str.strip().str.title()
    work["category_clean"] = normalize_categories(work["category"])
    critical = work[["invalid_date", "invalid_quantity", "invalid_unit_price", "invalid_discount", "invalid_labor_hours"]].any(axis=1) | work["category_clean"].isna()
    for row in work.loc[critical].itertuples():
        log.append({"row_identifier": row.transaction_id, "issue": "invalid critical field", "column": "validation flags", "action": "EXCLUDE", "reason": "cannot establish valid analytical observation"})
    work = work.loc[~critical].copy()
    work["date"] = work.pop("parsed_date")
    work["location"] = work.pop("location_clean")
    work["category"] = work.pop("category_clean")
    work["net_revenue"] = work["quantity"] * work["unit_price"] * (1 - work["discount"])
    work["outlier_candidate"] = outlier_candidates(work["net_revenue"])
    if "T027" in set(work.loc[work["outlier_candidate"], "transaction_id"]):
        log.append({"row_identifier": "T027", "issue": "extreme revenue candidate", "column": "net_revenue", "action": "RETAIN", "reason": "corroborated catering order; unusual is not invalid"})
    drop = ["invalid_date", "invalid_quantity", "invalid_unit_price", "invalid_discount", "invalid_labor_hours", "duplicate_id"]
    work = work.drop(columns=drop).sort_values("transaction_id").reset_index(drop=True)
    validate_cleaned(work)
    return work, audit, pd.DataFrame(log, columns=["row_identifier", "issue", "column", "action", "reason"])


def validate_cleaned(df: pd.DataFrame) -> None:
    """Fail loudly when the analytical dataset violates its contract."""
    assert df["transaction_id"].is_unique
    assert df[["date", "quantity", "unit_price", "discount", "category"]].notna().all().all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["quantity"].gt(0).all() and df["unit_price"].gt(0).all()
    assert df["discount"].between(0, 1).all() and df["labor_hours"].ge(0).all()
    assert set(df["category"]).issubset(EXPECTED_CATEGORIES)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); return path


def create_figures(raw: pd.DataFrame, cleaned: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create four plain Matplotlib quality and reconciliation views."""
    paths = []
    fig, ax = plt.subplots(figsize=(8, 4)); raw.isna().sum().plot.bar(ax=ax); ax.set(title="Missing values in raw evidence", ylabel="count"); paths.append(_save(fig, output_dir / "chapter-11-missingness.png"))
    before = raw["category"].fillna("<missing>").value_counts(); after = cleaned["category"].value_counts(); labels = sorted(set(before.index) | set(after.index)); fig, ax = plt.subplots(figsize=(9, 4)); x = range(len(labels)); ax.bar([i-.2 for i in x], [before.get(v, 0) for v in labels], .4, label="raw labels"); ax.bar([i+.2 for i in x], [after.get(v, 0) for v in labels], .4, label="cleaned labels"); ax.set_xticks(list(x), labels, rotation=35, ha="right"); ax.set(title="Category fragmentation before and after", ylabel="rows"); ax.legend(); paths.append(_save(fig, output_dir / "chapter-11-categories-before-after.png"))
    fig, ax = plt.subplots(figsize=(8, 3)); ax.boxplot(cleaned["net_revenue"], orientation="horizontal"); ax.scatter(cleaned.loc[cleaned["outlier_candidate"], "net_revenue"], [1] * cleaned["outlier_candidate"].sum(), color="tab:red", label="IQR candidate"); ax.set(title="Revenue outlier candidates are not automatic errors", xlabel="net revenue ($)"); ax.legend(); paths.append(_save(fig, output_dir / "chapter-11-outlier-candidate.png"))
    naive = (pd.to_numeric(raw["quantity"], errors="coerce") * pd.to_numeric(raw["unit_price"], errors="coerce") * (1-pd.to_numeric(raw["discount"], errors="coerce"))).sum(); fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["raw naive\n(available fields)", "clean analytical\n(documented decisions)"], [naive, cleaned["net_revenue"].sum()], color=["tab:gray", "tab:blue"]); ax.set(title="Different evidence produces different totals", ylabel="net revenue ($)"); paths.append(_save(fig, output_dir / "chapter-11-raw-vs-cleaned-revenue.png"))
    return paths


def run(output_dir: Path | None = None, processed_path: Path | None = None) -> int:
    """Run the inspect-to-document Chapter 11 experiment."""
    raw = load_chapter_11_data(); audit, structure = quality_audit(raw); flagged = add_quality_flags(raw)
    cleaned, _, log = clean_transactions(raw)
    processed = processed_path or PROJECT_ROOT / "data" / "processed" / "chapter-11-clean-transactions.csv"
    processed.parent.mkdir(parents=True, exist_ok=True); cleaned.to_csv(processed, index=False, date_format="%Y-%m-%d")
    log_path = processed.with_name("chapter-11-cleaning-log.csv")
    log.to_csv(log_path, index=False)
    paths = create_figures(raw, cleaned, output_dir or PROJECT_ROOT / "figures")
    naive = (pd.to_numeric(raw["quantity"], errors="coerce") * pd.to_numeric(raw["unit_price"], errors="coerce") * (1-pd.to_numeric(raw["discount"], errors="coerce"))).sum()
    print("Chapter 11 — Messy Data | inspect → detect → diagnose → decide → clean → validate → document")
    print(f"Raw evidence: shape={raw.shape}; grain=one recorded transaction row (including repeats)\n{raw.head(3).to_string(index=False)}")
    print("\nAudit:\n" + audit.to_string(index=False)); print("\nMissing by column:\n" + structure.loc[structure.missing_count.gt(0), ["missing_count", "missing_percent"]].round(1).to_string())
    print(f"Exact duplicates={raw.duplicated().sum()}; conflicting IDs={sorted(identify_conflicting_ids(raw))}")
    print("Raw location labels: " + ", ".join(map(str, raw["location"].value_counts().index)))
    print(f"Invalid date IDs={flagged.loc[flagged.invalid_date, 'transaction_id'].tolist()}; business-rule failures={int(flagged[['invalid_quantity','invalid_unit_price','invalid_discount','invalid_labor_hours']].any(axis=1).sum())}")
    print(f"Outlier candidates retained={cleaned.loc[cleaned.outlier_candidate, 'transaction_id'].tolist()}")
    print(f"Decisions: " + ", ".join(f"{k}={v}" for k, v in log.action.value_counts().items()))
    print(f"Reconciliation: raw rows={len(raw)}, clean rows={len(cleaned)}, excluded={len(raw)-len(cleaned)}; raw naive revenue=${naive:,.2f}, cleaned valid revenue=${cleaned.net_revenue.sum():,.2f}")
    print(f"Post-clean validation passed; processed={processed}; cleaning log={log_path}; figures={len(paths)}")
    print("Interpretation: totals differ because they include different evidence—not because historical performance changed. T027 is unusual but corroborated and retained.")
    print("Unresolved limitation: T032's conflicting source records remain excluded pending investigation. Raw data is evidence and was not overwritten.")
    return 0
