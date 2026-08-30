# Chapter 10 — Tables & DataFrames

> **Central question:** Why do analysts need DataFrames if NumPy arrays already exist?

Arrays give us numerical structure. **DataFrames give that structure labels, mixed data types, and business meaning.** Our workflow is:

> raw rows → labeled columns → DataFrame → inspect → select → filter → sort → derive columns → aggregate → preserve meaning

## Start with the table

| transaction_id | category | quantity | unit_price |
|---|---|---:|---:|
| T001 | Entree | 2 | 14.00 |
| T002 | Beverage | 3 | 4.50 |
| T003 | Dessert | 1 | 8.00 |

Each row is an **observation**, each column is a **variable**, and each column name states meaning. pandas also maintains an **index**, a row-label structure. A numerical subset can become a Chapter 9 matrix, but the full table mixes identifiers, dates, categories, and numbers that do not naturally belong in one homogeneous array.

Our clean fictional restaurant file has transaction grain: **one row = one transaction**. It intentionally has no missing values, duplicates, inconsistent labels, or malformed dates; those belong in Chapter 11.

## Load and inspect before analyzing

```python
import pandas as pd

df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])

df.head()     # What do representative rows look like?
df.shape      # How many rows and columns?
df.columns    # Which variables exist?
df.dtypes     # What types did pandas infer?
df.info()     # Optional compact structural report
```

Parsing changes `date` from text-like values into datetimes, enabling calendar operations. Business tables mix numbers, text/categories, dates, identifiers, and sometimes Booleans. `df.dtypes` shows that each column may have its own dtype—a decisive difference from a homogeneous NumPy array. Make this first-look sequence a habit.

## Series, DataFrame, columns, and index

```python
df["category"]                       # Series: one labeled dimension
df[["category", "net_revenue"]]     # DataFrame: two labeled dimensions
df["net_revenue"]                    # Series
df[["net_revenue"]]                  # one-column DataFrame
df.index
```

The inner brackets in the second form create a list of requested column names. Approximately, a Series is one labeled dimension and a DataFrame is two labeled dimensions; this analogy is useful, not a complete definition.

The default index labels happen to be `0, 1, ...`, but an index is not automatically a business identifier. Therefore:

```python
df.iloc[0]  # first row by position
df.loc[0]   # row whose index label is 0
```

They currently reach the same row only because label 0 occupies position 0. After sorting or assigning other labels they may differ. `df.set_index("transaction_id")` can help label rows by transaction, but is unnecessary for most work here.

Select rows and columns together by reading the two parts separately:

```python
df.loc[
    df["location"] == "Downtown",          # 1. choose rows
    ["date", "category", "net_revenue"], # 2. choose columns
]
```

## Boolean filtering and sorting

A comparison creates a labeled Boolean mask, connecting directly to Chapter 9:

```python
condition = df["net_revenue"] > 50
df[condition]

downtown_high_value = df[
    (df["location"] == "Downtown")
    & (df["net_revenue"] > 50)
]
```

Use `&` for and, `|` for or, and parentheses around every condition. An optional readable equivalent is `df.query("location == 'Downtown' and net_revenue > 50")`, but Boolean masks expose the core mechanism.

Which transactions were largest?

```python
df.sort_values("net_revenue", ascending=False)
df.sort_values(
    ["location", "net_revenue"],
    ascending=[True, False],
)
```

Sorting changes row order, not the meaning of a column or its values.

## Derived columns: named vectorized mathematics

For transaction (i),

\[
\text{gross}_i=q_i p_i
\]

\[
\text{net}_i=\text{gross}_i(1-d_i)
\]

Direct assignment makes those definitions executable:

```python
df["gross_revenue"] = df["quantity"] * df["unit_price"]
df["net_revenue"] = df["gross_revenue"] * (1 - df["discount"])
```

**A DataFrame column operation is still vectorized mathematics, but the variables now have meaningful names.** Alignment and labels preserve what each number represents. An optional expression is:

```python
df = df.assign(
    gross_revenue=lambda d: d["quantity"] * d["unit_price"]
)
```

Direct assignment remains our primary style. Names are part of analytical correctness: `df.rename(columns={"unit_price": "price_per_unit"})` can make intended meaning clearer. Cryptic names invite interpretation errors even when code runs.

## Dates and categories

```python
df["date"] = pd.to_datetime(df["date"])
df["date"].dt.day_name()
df["date"].dt.month
df["day_of_week"] = df["date"].dt.day_name()
df["category"].value_counts()
```

Calendar accessors create useful business features without turning this into formal time-series analysis. Frequency counts give a first view of categorical variables such as category, location, and customer type; encoding categories is a later topic.

## Aggregation and `groupby`

Chapter 3's summation appears directly:

```python
df["net_revenue"].sum()
df["net_revenue"].mean()
df["quantity"].sum()
```

To answer “What is total revenue by category?” split observations into groups → perform the same aggregation within each group → combine results:

```python
df.groupby("category")["net_revenue"].sum()
df.groupby("location")["net_revenue"].mean()
```

Named aggregation creates a compact location-grain business table:

```python
summary = (
    df.groupby("location")
      .agg(
          transactions=("transaction_id", "count"),
          units=("quantity", "sum"),
          revenue=("net_revenue", "sum"),
          average_transaction=("net_revenue", "mean"),
      )
)
summary.reset_index()
```

Each output name is followed by `(source_column, operation)`. Grouping variables become index labels by default; `reset_index()` moves `location` back to an ordinary column and supplies a fresh default index.

One location can have highest total revenue while another has highest average transaction value. There is no contradiction: total reflects transaction volume as well as size, while the average divides revenue by transaction count. **Total, average, count, and rate answer different questions.**

## Grain: what does one row represent right now?

The **grain** tells us what one row represents:

- source data: one row = one transaction;
- daily aggregation: one row = one day;
- location aggregation: one row = one location.

Always ask **“What does one row represent right now?”** before interpreting a metric. Transaction-level revenue cannot simply be compared or combined as though it had the same grain as location-level averages. Doing so can duplicate values or confuse denominators. We keep that warning conceptual; joins begin in Chapter 12.

## Moving to NumPy

```python
X = df[["quantity", "unit_price", "discount"]].to_numpy()
```

Keep a DataFrame while labels, mixed types, selection, and business interpretation matter. Move a deliberately selected numeric block to NumPy for lower-level numerical work. Moving between them is normal; neither is universally superior. The values agree, but `X` no longer carries the three column names, so preserve or document their order.

## Common pandas pitfalls

1. **Series versus DataFrame:** `df["x"]` is a Series; `df[["x"]]` is a one-column DataFrame.
2. **Python `and`:** `condition_a and condition_b` asks for single truth values. Use `condition_a & condition_b` for element-wise Series logic.
3. **Missing parentheses:** `df["x"] > 5 & df["y"] < 10` is wrong or ambiguous. Write `(df["x"] > 5) & (df["y"] < 10)`.
4. **Chained assignment:** avoid `df[df["x"] > 5]["flag"] = True`. State rows and destination together: `df.loc[df["x"] > 5, "flag"] = True`.

## Executable experiment and visualizations

Run:

```bash
python3 -m analytics_foundations chapter-10
```

The experiment loads and inspects 24 clean transaction rows, parses dates, derives gross/net revenue and day of week, filters and sorts high-value Downtown transactions, calculates overall and grouped metrics, identifies category and location leaders, reconciles grouped revenue with the total, and produces:

1. net revenue by category: raw rows → `groupby` → summary table → bar chart;
2. total net revenue by location;
3. daily aggregated revenue, changing grain from transaction to day.

## Mastery checkpoints

### Concept checkpoint

1. What does one row represent?
2. What differs between a Series and DataFrame?
3. What differs between `.loc` and `.iloc`?
4. Why are column labels valuable?
5. What does `groupby` conceptually do?
6. What does dataset grain mean?
7. Why might total revenue and average transaction value rank locations differently?

### Execution checkpoint

Select three columns; filter one location; filter above a revenue threshold; sort descending; derive a margin or revenue column; calculate grouped totals and averages; create day of week; and convert selected numeric columns to NumPy. Confirm category totals sum to the overall total.

### Interpretation checkpoint

**Location A has more total revenue than Location B, but B has higher average transaction value. Which performs better?** Those metrics alone cannot decide. The objective and customer volume, labor, margin, capacity, and other measures may matter.

**Why know grain before calculating metrics?** Sums, averages, and counts derive their meaning and denominators from what each row represents. A valid formula at the wrong grain can yield a misleading claim.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** correctly structured observations, variables, grain, filtering, and grouping help prevent statistical misinterpretation.
- **BUAD 512B — Business Modeling with Python:** pandas DataFrames support loading, inspecting, selecting, transforming, and summarizing business data.
- **BUAD 5112 — Competing Through Business Analytics:** metrics become useful only when observations, denominators, categories, and business meaning are understood.

This independent preparation chapter does not reproduce any William & Mary course.
