# Chapter 0 — The Analytics Laboratory

This chapter follows one small investigation from question to evidence. It is a map of the
journey ahead, not a statistics lesson. Our fictional **Riverside Cafe** recorded two weeks
of daily operations and asks:

> **Which days appear strongest, and what evidence supports that conclusion?**

## Learning objectives

By the end, you can explain the basic analytics workflow; distinguish a business question
from a computational task; recognize observations and variables; load and inspect a CSV;
calculate and group descriptive measures; make a plot; translate a result into business
language; avoid turning a pattern into a causal claim; and explain reproducibility.

## 1. The business problem

“Which days appear strongest?” is a **business question**: it points toward a decision but
does not tell a computer what to do. “Group rows by `day_of_week` and calculate mean
`revenue`” is a **computational task**. Before calculating, we must decide what “strongest”
means. Revenue captures sales volume, while revenue per labor hour captures one kind of
operating productivity. A useful investigation examines both rather than silently choosing.

Our miniature workflow is:

> business question → inspect observations → identify variables → calculate metrics →
> visualize → compare → interpret → state limitations → ask the next question

## 2. Meet the data

The file [`data/raw/chapter-00-cafe-daily.csv`](../../data/raw/chapter-00-cafe-daily.csv)
contains 14 clean daily records. A **dataset** is a structured collection of data. Each row
is an **observation**—one day at the cafe. Each column is a **variable**—a characteristic
recorded for every day:

| Variable | Meaning | Kind of value |
|---|---|---|
| `date` | calendar date | date |
| `day_of_week` | weekday name | categorical text |
| `customers` | customer count | numeric count |
| `revenue` | daily sales in dollars | numeric measure |
| `labor_hours` | total staff hours | numeric measure |

This deliberately clean data lets us study the workflow. Missing values, duplicates, and
inconsistent categories wait until Chapter 11.

## 3. Ask before calculating

An analytical question is specific enough to connect a business concern to evidence. Here
we ask which observed dates and weekdays have greater revenue, customer traffic, and labor
productivity. A **metric** is a defined numerical summary used for comparison. Its definition
matters: “average revenue per customer” means total revenue divided by total customers,
not the unweighted average of 14 daily ratios.

Before touching code, check the unit of observation, variable meanings, time span, and which
metric serves the decision. Revenue is not profit, and two weeks may not represent a season.

## 4. First measurements

We will calculate:

- total revenue: sum of daily revenue;
- average daily revenue: total revenue divided by observed days;
- total customers: sum of daily customer counts;
- average revenue per customer: total revenue divided by total customers; and
- revenue per labor hour: total revenue divided by total labor hours.

These are **calculations**. Their business meaning comes later through interpretation.

## 5. The mathematics underneath

The arithmetic mean of daily revenue is

```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
```

Read this as “add every daily revenue and divide by the number of days.” Here, \(x_i\) is
the revenue for day \(i\); \(i\) labels each day from the first through the last; \(n\) is
the number of observed days; \(\sum\) means “add”; and \(\bar{x}\) (x-bar) is their mean.
You only need to recognize the notation now; Chapter 3 develops summation carefully.

## 6. Hand-worked example

Start with the first three revenues—$667.00, $750.20, and $995.40—before using a shortcut:

```text
sum = 667.00 + 750.20 + 995.40 = 2,412.60
mean = 2,412.60 / 3 = 804.20
```

The hand calculation exposes what the mean does. Pandas applies precisely that operation
to all observations:

```python
first_three = data["revenue"].head(3)
hand_equivalent = first_three.sum() / len(first_three)
library_shortcut = first_three.mean()
```

## 7. Make it executable

From the repository root, load and inspect rather than dumping the whole table:

```python
import pandas as pd

data = pd.read_csv("data/raw/chapter-00-cafe-daily.csv", parse_dates=["date"])
print(data.head())
print(data.shape)
print(data.dtypes)
```

Then connect definitions to code:

```python
total_revenue = data["revenue"].sum()
average_daily_revenue_from_definition = total_revenue / len(data)
average_daily_revenue_with_pandas = data["revenue"].mean()

total_customers = data["customers"].sum()
revenue_per_customer = total_revenue / total_customers
revenue_per_labor_hour = total_revenue / data["labor_hours"].sum()
```

Group rows sharing a weekday and compare their means:

```python
by_weekday = data.groupby("day_of_week", sort=False).agg(
    average_revenue=("revenue", "mean"),
    average_customers=("customers", "mean"),
)
```

The source experiment remains intentionally visible in
[`src/analytics_foundations/chapter_00.py`](../../src/analytics_foundations/chapter_00.py).

## 8. See the pattern

A line chart preserves date order, showing both level and day-to-day variation:

```python
import matplotlib.pyplot as plt

data.plot(x="date", y="revenue", marker="o", legend=False)
plt.title("Riverside Cafe revenue by date")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.show()
```

A chart does not replace a metric. It helps us see whether a summary hides variation and
whether high values repeat. The CLI saves the reproducible version to
`figures/chapter-00-revenue-by-date.png`.

## 9. Interpret the evidence

**Calculation:** Saturday has the highest average revenue among the weekdays observed.

**Interpretation:** Weekend demand may be stronger in these two weeks. Compare customer
counts and revenue per labor hour before treating high sales as broad operating strength.
Interpretation translates a numerical result into cautious, decision-relevant language;
it should never claim more than the data supports.

## 10. What we cannot conclude

An observed pattern is not automatically causal. The data cannot establish that *being
Saturday causes* revenue to rise. Weather, promotions, holidays, opening hours, menu mix,
or chance could differ too. We also lack costs, so revenue cannot establish profitability.
Fourteen observations cover only two examples of each weekday and give no measure of how
stable this pattern is. Formal statistical inference comes later.

Useful additions include more weeks, promotion indicators, weather, product-level sales,
opening hours, and labor cost. Reproducibility matters because saved source data and code
let another analyst repeat the same definitions, detect mistakes, regenerate the figure,
and update the result when new observations arrive.

## 11. Experiment

Install the project, then run:

```bash
python3 -m analytics_foundations chapter-00
```

The command previews five observations, prints core metrics and weekday comparisons,
saves the figure, and ends with an interpretation and next question. Re-running the same
command from the same versioned inputs reproduces the analytical path.

## 12. Mastery checkpoints

### Concept checkpoint

1. What does one observation represent, and which fields are variables?
2. How is a metric different from an analytical question?
3. Separate this sentence into calculation and interpretation: “Saturday averaged the
   most revenue, suggesting weekend demand may be stronger.”

### Execution checkpoint

1. Without copying the experiment, calculate customers per labor hour overall.
2. Identify the weekday with the highest revenue per customer. State exactly how you
   aggregate the numerator and denominator.
3. Create a labeled scatter plot of customers versus revenue and explain what each point is.

### Interpretation checkpoint

1. If Saturday has the highest average revenue, can we conclude that Saturday causes higher
   revenue? Why not?
2. Does the highest-revenue weekday necessarily have the best labor productivity?
3. Name one omitted variable and explain how collecting it would improve the investigation.

## 13. Where we go next

The next analytical question is: **Are weekends consistently stronger over a longer
period, and does higher revenue merely reflect more customers or better efficiency?** The
remaining chapters develop the mathematical, programming, data, probability, and
statistical tools needed to answer such questions responsibly. Chapter 0 stops at a small
descriptive investigation; it does not perform inference.
