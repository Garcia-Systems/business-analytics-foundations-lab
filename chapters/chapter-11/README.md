# Chapter 11 — Messy Data

> **Central question:** Before analyzing a dataset, how do we determine whether the observations can be trusted?

Every prior dataset was intentionally clean. This fictional restaurant extract is not. Its grain is **one recorded transaction row**, not necessarily one trustworthy transaction: a retry can repeat a row and two records can claim the same identifier. Our workflow is:

> inspect → detect → diagnose → decide → clean → validate → document

**Data cleaning is not cosmetic. Every decision changes the evidence available for analysis.** Inspection therefore comes before correction.

## Learning objectives

By the end, you can distinguish cleaning from analysis; inspect structure, missingness, duplicates, identifiers, categories, dates, types, rules, and extremes; distinguish outliers from errors; choose `CORRECT`, `STANDARDIZE`, `IMPUTE`, `FLAG`, `EXCLUDE`, or `RETAIN`; preserve an audit trail; validate a processed file; reconcile raw and clean metrics; and explain how cleaning may introduce bias.

## Start with raw evidence

```python
raw = pd.read_csv("data/raw/chapter-11-messy-transactions.csv")
raw.head()
raw.shape
raw.columns
raw.dtypes
raw.info()
```

Ask **what looks suspicious before calculating a business metric?** The raw file stays in `data/raw/`; code creates a copy and reproducibly writes `data/processed/chapter-11-clean-transactions.csv`. It never edits the source. **Raw data is evidence. Preserve it.**

The 42 recorded rows include mixed date representations, a nonnumeric `quantity` token, missing values, fragmented labels, impossible values, an exact repeat, a conflicting identifier, and a large catering order. A dtype problem is often a symptom: coercing `"five"` detects that it cannot safely become a number; conversion alone does not determine the intended value.

## Audit first, then inspect the rows

```python
raw.isna()              # cell-level mask
raw.isna().sum()        # count by column
raw.isna().mean()       # proportion by column
raw.duplicated().sum()
raw["transaction_id"].duplicated(keep=False)
raw["location"].value_counts(dropna=False)
```

The reusable audit reports row/column counts, missing count and percentage, unique counts, exact repeats, repeated-ID rows, conflicting IDs, invalid dates, and numeric-rule failures. Numeric conversion also enables relevant minima and maxima during diagnosis. A summary points toward evidence; it must not replace inspection of the underlying rows.

Missing, zero, an empty string, and the literal label `"Unknown"` differ. Missing may mean never collected, not applicable, lost in integration, or an informative workflow failure. Ask whether another field can safely reconstruct it. `raw.dropna()` is available but blindly removes any row with any missing field. Even targeted removal:

```python
raw.dropna(subset=["quantity", "unit_price"])
```

is an analytical decision because those fields are essential to transaction revenue. T008's price is reconstructable from documented fields:

\[
\text{unit price}=\frac{\text{gross revenue}}{\text{quantity}}=\frac{100}{5}=20.
\]

That correction is valid only under the stated business rule that gross revenue equals quantity times price. Missing customer type is noncritical and becomes `Unknown`; this retains the transaction without pretending it was a member or guest. Arbitrary mean filling would change a numerical distribution merely because pandas makes it easy.

## Repeats and identifiers are different

```python
raw.duplicated()
raw.drop_duplicates()
raw[raw["transaction_id"].duplicated(keep=False)]
```

The second T020 row is identical, so excluding that extra record is defensible. T032 appears twice with different prices. This is not solved by automatically keeping first or last: both source records are flagged and excluded from analytical results pending investigation. **Exact duplicate rows and duplicate business keys are not the same problem.**

## Syntax, semantics, dates, and types

```python
location_clean = raw["location"].str.strip().str.lower().str.title()
category_key = raw["category"].str.strip().str.lower()
category_map = {"drinks": "Beverage", "beverage": "Beverage"}
category_clean = category_key.map(category_map)
```

Stripping whitespace and standardizing case fix syntactic differences such as `Downtown ` and `DOWNTOWN`. Mapping `Drinks` to `Beverage` is semantic: it needs approved business knowledge, not just lowercasing. The pipeline keeps the untouched DataFrame and exposes normalized columns while deciding.

```python
parsed_date = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
raw.loc[parsed_date.isna()]
quantity = pd.to_numeric(raw["quantity"], errors="coerce")
```

Recognizable representations become timestamps; impossible `2026-13-05` becomes `NaT`. Coercion detects uncertainty—it does not repair it. The source's different date formats are parseable inconsistency, while the impossible month is invalid evidence.

## Business-rule validation

Statistical tools cannot determine every invalid value. Business constraints add explicit rules:

\[
q>0,\quad p>0,\quad 0\le d\le1,\quad h\ge0.
\]

```python
invalid_quantity = quantity.isna() | quantity.le(0)
invalid_price = unit_price.notna() & unit_price.le(0)
invalid_discount = discount.isna() | ~discount.between(0, 1)
invalid_labor = labor_hours.isna() | labor_hours.lt(0)
```

The pipeline exposes `invalid_date`, `invalid_quantity`, `invalid_unit_price`, `invalid_discount`, `invalid_labor_hours`, and `duplicate_id` flags before it decides. A negative quantity might represent a return in another system; a 150% discount might be a percent/decimal integration mistake. We first ask what it could mean. Here, without corroborating business rules for returns or corrections, unresolved critical rows are logged and excluded rather than silently rewritten.

## Outliers are investigation candidates

After valid revenue is calculated, sort it:

```python
cleaned.sort_values("net_revenue", ascending=False)
```

For the IQR screen,

\[
IQR=Q_3-Q_1,
\]

flag values below \(Q_1-1.5IQR\) or above \(Q_3+1.5IQR\). Call the result an **outlier candidate**, not bad data. Discount `1.5` violates an explicit rule; T027's $1,250 catering sale is unusual but possible, supported by quantity, gross revenue, category, customer type, and labor hours, and retained. Magnitude alone never proves error.

Ask: Was it a catering or group order? Could it be a decimal error? Did the system allow the amount? Does another field corroborate it? Several modest values may also cross a tight IQR boundary; flags initiate investigation rather than automatic deletion.

## Decide and document

For every issue ask:

1. What is wrong or uncertain?
2. Is it syntactic or semantic?
3. Can the correct value be established?
4. Could removal bias the analysis?
5. Should it be corrected, standardized, imputed, flagged, excluded, or retained?
6. How will that decision be documented?

The machine-readable cleaning log records `row_identifier`, `issue`, `column`, `action`, and `reason`. T008 is `CORRECT`; label variants are `STANDARDIZE`; T014 is `IMPUTE`; invalid critical rows and unresolved T032 are `EXCLUDE`; corroborated T027 is `RETAIN`. This is a readable audit trail, not a heavyweight governance system.

Post-clean assertions require unique IDs, complete critical fields, parsed dates, positive quantities/prices, bounded discounts, nonnegative labor, and approved categories. They fail loudly if the deterministic pipeline stops satisfying its contract. Re-running the pipeline produces the same rows and values.

## Reconciliation, bias, and business reality

The naive raw total and validated analytical total cover different observations. Neither label makes a total automatically correct. A lower cleaned total does not mean cleaning reduced historical performance: **the dataset changed, not the business history**.

Suppose missing revenue occurs disproportionately on busy days. Complete-case filtering can lower apparent average revenue by changing which transactions remain. A complete table is not necessarily an unbiased sample. Cleaning decisions change the represented population.

Mess also diagnoses business systems: inconsistent location names may reveal several sources, duplicate IDs may reveal retry logic, missing fields may expose broken workflows, and invalid discounts may reveal weak point-of-sale validation. Technical transformations require defensible business rules.

## Executable experiment and figures

```bash
python3 -m analytics_foundations chapter-11
```

The command inspects raw evidence, prints a concise audit and implicated identifiers, applies documented decisions, writes the processed CSV and log in memory, validates the result, reconciles totals, states the unresolved T032 limitation, and creates:

1. missing-value counts;
2. raw versus standardized category counts;
3. a revenue boxplot with IQR candidates;
4. naive raw versus documented clean revenue (different evidence, not “improvement”).

## Mastery checkpoints

### Concept checkpoint

Why preserve raw data? How does an exact repeat differ from a repeated identifier? Why is missing not zero? Why is an outlier not automatically wrong? How do syntactic and semantic cleaning differ? Why can dropping rows bias analysis? Why are business rules necessary?

### Inspection checkpoint

Diagnose—do not yet clean—this table:

| id | date | location | category | quantity | price | discount |
|---|---|---|---|---:|---:|---:|
| X1 | 2026-10-01 | Wharf | Drinks | 2 | 5 | 0 |
| X1 | 2026-10-01 | Wharf | Beverage | 2 | 5.5 | 0 |
| X2 | 2026-02-30 | wharf  | Entree | -1 | 18 | 0 |
| X3 | 10/03/2026 | Wharf | Dessert | 50 | 25 | 0.1 |
| X3 | 10/03/2026 | Wharf | Dessert | 50 | 25 | 0.1 |
| X4 | 2026-10-04 | Wharf |  | 2 | 8 | 1.4 |

Identify missingness, an exact repeat, a conflicting identifier, category fragmentation, invalid date, impossible values, and a suspicious but potentially legitimate extreme.

### Execution checkpoint

Calculate missing counts; inspect category variants; identify both duplicate types; parse dates; create at least two rule flags; standardize one category with an approved mapping; retain one corroborated outlier candidate; and generate a validation report.

### Interpretation checkpoint

**A transaction is ten times larger than most. Remove it?** Not from magnitude alone; investigate and validate against business context.

**Removing every row with any missing field makes the table complete. Is analysis unbiased?** Not necessarily. Complete-case filtering may systematically alter which observations remain.

**Cleaned total revenue is lower. Did cleaning reduce performance?** No. Transformations changed included evidence, not historical performance.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** inference depends on supplied observations; missing, repeated, invalid, or selective exclusions can change estimates.
- **BUAD 512B — Business Modeling with Python:** wrangling needs systematic inspection, parsing, normalization, filtering, validation, and reproducible transformations.
- **BUAD 5112 — Competing Through Business Analytics:** quality problems reveal process and system weaknesses; reliable metrics need technical transformations and defensible rules.

This independent preparation chapter does not reproduce any William & Mary course.
