# Chapter 12 — Transform, Group & Join

> **Central question:** How do we combine data from different business processes without accidentally changing what the rows mean?

The fictional restaurant group asks which locations generate the strongest revenue relative to labor input. Revenue, labor, and readable location labels live in separate operational tables. The workflow is:

> grain → keys → transform → group → aggregate → join → validate → interpret

**Joining tables is not merely matching columns. It changes the structure of the evidence, so analysts must understand keys, cardinality, and grain before combining data.**

## Learning objectives

By the end, you can explain grain; identify primary, foreign, and composite keys; distinguish one-to-one, one-to-many, many-to-one, and many-to-many relationships; group and aggregate deliberately; choose inner, left, right, or outer joins; validate cardinality and match coverage; detect row multiplication; reconcile totals; and preserve the meaning of one row.

## Begin with grain and keys

```text
transactions: one row = one transaction
locations:    one row = one location
labor_daily:  one row = one location-day
```

Can they be joined directly without changing grain? **Not always.** Grain says what one row means and governs every operation. A **primary key** identifies one row uniquely within its table: `transactions.transaction_id` and `locations.location_id`. A **foreign key** references an entity elsewhere: `transactions.location_id` references `locations.location_id`. These are analytical concepts here, not a lesson in database constraints.

```python
transactions["transaction_id"].duplicated().sum()
locations["location_id"].is_unique
```

The first must be zero and the second true. Yet `transactions.location_id` legitimately repeats because one location has many transactions. Duplicates are only errors when they violate the table's grain.

Locations → transactions is **one-to-many**; viewed in reverse, transactions → locations is **many-to-one**. A one-to-one relationship has one matching row on each side. A many-to-many relationship has repeated matching keys on both sides and creates combinations.

## A safe metadata merge

```python
with_names = transactions.merge(
    locations,
    on="location_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)
with_names["_merge"].value_counts()
```

For each transaction, this attaches its location. Because the right key is unique, transaction row count should remain unchanged. `validate` makes the cardinality assumption executable: pandas also supports `"one_to_one"`, `"one_to_many"`, and `"many_to_many"`. Do not use many-to-many casually to silence an error. A merge should not merely run; inspect `_merge` for `left_only`, `right_only`, and `both`. Missing keys cannot match and require attention.

Join types are business decisions:

| left keys | right keys | inner | left | right | outer |
|---|---|---|---|---|---|---|
| A, B | B, C | B | A, B | B, C | A, B, C |

An **inner** join retains shared matches; **left** retains the analytical population on the left; **right** retains the right population; **outer** retains evidence from both. If transactions define the population and locations merely describe it, a left join avoids hiding activity with missing metadata—though unmatched definitions still require investigation. Join type is an analytical decision, not a syntax preference. When non-key names overlap, explicit `suffixes=("_transaction", "_location")` prevents ambiguity.

## Composite keys and the dangerous join

Labor is one row per pair

\[
(\text{location},\text{date}).
\]

Neither column is unique alone; together they are a **composite key**:

```python
labor_daily.duplicated(subset=["location_id", "date"]).sum()
```

Joining daily labor onto transactions is legal and useful for some row-level questions:

```python
transactions.merge(labor_daily, on=["location_id", "date"], validate="many_to_one")
```

But it repeats a day's labor on every transaction. If one location-day has four transactions and 20 labor hours, its rows contain 20, 20, 20, 20. Naively summing gives

\[
20+20+20+20=80,
\]

not the actual 20. The business did not use more labor; a mismatched-grain join duplicated the denominator.

## Aggregate before joining

Transform toward the grain required by the question:

```python
daily_revenue = (
    transactions
    .groupby(["location_id", "date"], as_index=False)
    .agg(
        revenue=("net_revenue", "sum"),
        transactions=("transaction_id", "count"),
    )
)

daily = daily_revenue.merge(
    labor_daily,
    on=["location_id", "date"],
    how="left",
    validate="one_to_one",
)
daily = daily.assign(
    revenue_per_labor_hour=lambda x: x.revenue / x.labor_hours,
    day_of_week=lambda x: x.date.dt.day_name(),
)
```

`as_index=False` keeps grouping variables as ordinary columns, which simplifies later merges. Both inputs now have location-day grain. `.assign`, parsed date types, and named aggregation make the transformation intent readable.

A pivot is another structured aggregation and presentation:

```python
pd.pivot_table(daily, index="location_id", columns="day_of_week", values="revenue", aggfunc="sum")
```

It changes the presentation of aggregated data; pivoting is not the core issue.

Next move deliberately to location grain and attach labels:

```python
location_summary = (
    daily.groupby("location_id", as_index=False)
    .agg(revenue=("revenue", "sum"), labor_hours=("labor_hours", "sum"))
    .assign(revenue_per_labor_hour=lambda x: x.revenue / x.labor_hours)
    .merge(locations, on="location_id", how="left", validate="one_to_one")
)
```

The rate is evidence about revenue relative to labor input, not an automatic judgment of service quality, workload, capacity, labor mix, profitability, or demand.

## Ratio of sums, not accidental average of ratios

Day 1 has revenue 100 and labor 10, so its rate is 10. Day 2 has revenue 900 and labor 30, so its rate is 30. The average day's rate is

\[
(10+30)/2=20,
\]

while overall revenue per labor hour is

\[
(100+900)/(10+30)=25.
\]

Both are valid but answer different questions. The first weights days equally; the second weights daily rates by labor hours and reconnects to Chapter 3's weighted-average lesson.

## Many-to-many multiplication

| customer_id | order |
|---|---|
| C1 | A |
| C1 | B |

joined to

| customer_id | campaign |
|---|---|
| C1 | Email |
| C1 | SMS |

produces A–Email, A–SMS, B–Email, and B–SMS: \(2\times2=4\) combinations. This can be intended, but often reveals an undefined target grain. Before and after joins inspect `len(left)`, `len(right)`, and `len(merged)`—then also check unique keys, totals, match coverage, and cardinality because row count alone is insufficient.

## Reconciliation and the grain ledger

| object | grain |
|---|---|
| transactions | transaction |
| daily_revenue | location-day |
| labor_daily | location-day |
| daily_metrics | location-day |
| location_summary | location |

Use floating-point-tolerant checks to establish

\[
\text{transaction revenue}=\text{daily revenue}=\text{location revenue}
\]

and raw labor = compatible-grain daily labor = location labor. Every transformation should preserve or deliberately redefine measurable totals.

## Hand-worked workflow

1. Identify the three source grains above.
2. Identify transaction and location primary keys and the transaction location foreign key.
3. State the one-to-many locations-to-transactions relationship.
4. Verify labor's `(location_id, date)` composite key.
5. Observe repeated labor after the transaction-level join.
6. Group revenue to location-day before the one-to-one labor join.
7. Confirm the 2×2 many-to-many example yields four rows.
8. Distinguish average daily ratios (20) from ratio of sums (25).
9. Reconcile transaction, daily, and location revenue totals.

## Executable experiment and figures

```bash
python3 -m analytics_foundations chapter-12
```

The experiment loads and reports all grains; validates keys; audits a many-to-one metadata join; isolates the denominator-duplication trap; constructs and validates daily and location metrics; reconciles revenue and labor; demonstrates 2×2 multiplication; and creates:

1. a transaction-to-location-day grain schematic;
2. separate revenue and labor panels by location;
3. revenue per labor hour by location.

## Mastery checkpoints

### Concept checkpoint

What is grain? What is a key? Why may `location_id` repeat in transactions? What is a composite key? What does one-to-many mean? Why can a join increase row count? Why is many-to-many dangerous? Why aggregate before joining? What does `validate="many_to_one"` protect against?

### Grain checkpoint

```text
orders:      one row = order
order_items: one row = product within order
customers:   one row = customer
```

Which keys should be unique? What relationship connects orders and items? If order-level revenue is joined to items and summed, what happens? Expected reasoning: order ID is unique in orders, customer ID in customers, `(order_id, product/item identifier)` identifies items, orders-to-items is one-to-many, and repeated order revenue becomes overstated.

### Execution checkpoint

Inspect uniqueness; perform and validate a many-to-one merge; inspect unmatched rows; aggregate transactions; merge on a composite key; calculate a rate; and reconcile totals across grains.

### Interpretation checkpoint

**After joining daily labor onto every transaction, total labor increased. Did the business use more labor?** No. The same daily labor was repeated because grains differed.

**Location A has the highest average daily rate. Must it have the highest overall rate?** No. A simple average of daily ratios can differ from the ratio of total revenue to total labor.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** statistical analysis assumes correctly defined observations; accidental duplication can violate that assumption before modeling.
- **BUAD 512B — Business Modeling with Python:** real work requires transforming, grouping, joining, validating, and reconciling multiple sources.
- **BUAD 5112 — Competing Through Business Analytics:** operational systems must be combined correctly to avoid polished but invalid metrics.

This independent preparation chapter does not reproduce any William & Mary course.
