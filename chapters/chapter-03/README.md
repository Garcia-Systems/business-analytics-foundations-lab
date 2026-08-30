# Chapter 3 — Summation & Aggregation

> **Central outcome:** Summation notation is not mysterious mathematical decoration. It is a compact instruction: perform a calculation across observations and add the results. Many everyday business metrics—and much of statistics—are built from that idea.

Chapter 2 distinguished repeated addition from repeated multiplication. This chapter asks how **Riverside Cafe** can combine individual transactions to answer: How much revenue did the cafe generate? What was average revenue per transaction? Which category contributed most? What was the average price actually paid per unit?

## Learning objectives

By the end, you should be able to read an index and summation bounds; expand a sum and write repeated addition with sigma notation; calculate totals and arithmetic means; distinguish simple and weighted averages; translate sums into Python, NumPy, and pandas; aggregate within groups; calculate contribution to a total; and recognize the instruction represented by summation in later statistics formulas.

## 1. From observations to sigma notation

Suppose four observed values are

\[
x_1=12,\quad x_2=18,\quad x_3=15,\quad x_4=20.
\]

A subscript identifies an observation: $x_3$ means the value attached to observation 3, not $x\times3$. Repeated addition is

\[
x_1+x_2+x_3+x_4=12+18+15+20=65.
\]

The compact version is

\[
\sum_{i=1}^{4}x_i=x_1+x_2+x_3+x_4.
\]

Read every piece as an instruction:

- $\Sigma$, the Greek capital sigma, says **add the terms**.
- $i$ is the **index**, a counter identifying an observation.
- $i=1$, the lower bound, says start at observation 1.
- $4$, the upper bound, says stop at observation 4.
- $x_i$ says use the value at observation $i$.

For more practice,

\[
\sum_{i=2}^{4}x_i=x_2+x_3+x_4=18+15+20=53,
\]

and

\[
\sum_{i=1}^{3}2x_i=2x_1+2x_2+2x_3=24+36+30=90.
\]

The general form

\[
\sum_{i=1}^{n}x_i=x_1+x_2+\cdots+x_n
\]

uses $n$ for the number of observations. In a dataset, **each row can be thought of as observation $i$**. The index links notation directly to rows rather than introducing an abstract puzzle.

## 2. Transactions become total revenue

The clean source data contain 15 cafe transactions with an ID, day, category, quantity, and unit price. Revenue is deliberately calculated rather than stored. For transaction $i$,

\[
r_i=q_ip_i,
\]

where $q_i$ is quantity, $p_i$ is unit price, and $r_i$ is transaction revenue. For three tiny transactions with ($(q,p)$=(2,10),(3,4),(1,15)),

\[
r_1=2(10)=20,\quad r_2=3$4$=12,\quad r_3=1(15)=15,
\]

so

\[
R=\sum_{i=1}^{3}r_i=20+12+15=47.
\]

Substituting the row calculation gives the equivalent expression

\[
R=\sum_{i=1}^{n}q_ip_i.
\]

It says: multiply quantity by price for each row, then add those row results.

## 3. One sum, three implementations

An explicit loop exposes accumulation:

```python
total = 0
for value in revenues:
    total += value
```

NumPy expresses the same instruction over an array:

```python
np.sum(revenues)
```

pandas applies it to a labeled column:

```python
df["revenue"].sum()
```

All three implement $\sum r_i$. The loop is useful pedagogy, not inherently bad. NumPy and pandas operations are usually clearer for analytical work because they state the intended aggregation directly and operate naturally on arrays and tables.

## 4. The arithmetic mean is an aggregation

The sample mean is

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
       =\frac{x_1+x_2+\cdots+x_n}{n}.
\]

Here $x_i$ is each observation, the sum creates the total, $n$ counts observations, division distributes the total evenly, and $\bar{x}$ names the sample mean. For revenues 20, 12, and 15,

\[
\bar r=\frac{20+12+15}{3}=\frac{47}{3}\approx15.67.
\]

These are different languages for the same operation:

```python
sum(values) / len(values)  # Python
np.mean(values)            # NumPy
df["revenue"].mean()      # pandas
```

Average transaction revenue divides revenue by transactions. It does **not** divide by units; that would answer another question.

## 5. The correct average depends on what is being averaged

Imagine one transaction sells 1 cake at $20 and another sells 9 drinks at $4. A simple average of the two listed prices is

\[
\frac{20+4}{2}=12.
\]

But customers bought ten units, nine at the lower price. The quantity-weighted average is

\[
\bar p_w=\frac{\sum_{i=1}^{n}q_ip_i}{\sum_{i=1}^{n}q_i}
=\frac{1(20)+9$4$}{1+9}=\frac{56}{10}=5.60.
\]

Here $p_i$ is price and $q_i$ is its quantity or weight. The numerator is total revenue, the denominator is total units, and the result is average revenue per unit. The $12 simple average describes two price observations equally; $5.60 describes the price attached to the average unit sold. Neither operation is magic—the correct denominator and weights depend on the business question.

## 6. Grouped summation and contribution

Total revenue asks for all rows. Revenue for category $g$ is

\[
R_g=\sum_{i\in g}r_i.
\]

Read $i\in g$ gently: **add only observations belonging to group $g$**. In pandas,

```python
df.groupby("category")["revenue"].sum()
```

Thus

\[
\text{grouped summation}\quad\longleftrightarrow\quad\texttt{groupby(...).sum()}.
\]

The first chart follows this pipeline directly:

> transaction rows → grouped revenue sums → bar heights

A category's share and percentage contribution are

\[
\text{share}_g=\frac{R_g}{R},\qquad
\text{percentage}_g=100\times\frac{R_g}{R}.
\]

The shares should sum to 1 and the percentages to 100%, apart from harmless floating-point rounding. Group totals should also reconcile to overall revenue: $\sum_gR_g=R$.

Revenue leadership is not automatically price or quantity leadership. A category can produce large revenue because it sells many units, charges high prices, or combines both. Always name precisely what a metric aggregates.

### A short cumulative extension

A running total through observation $k$ is

\[
S_k=\sum_{i=1}^{k}x_i.
\]

In pandas, `df["revenue"].cumsum()` accumulates revenue row by row. This is useful when sequence matters, but the chapter figures remain focused on grouped sums and contributions.

## 7. Run the experiment

From the repository root:

```bash
python3 -m analytics_foundations chapter-03
```

The experiment loads the transactions, calculates row revenue, previews five rows, verifies loop/NumPy/pandas totals, reports overall and grouped metrics, creates two figures, and gives a concise interpretation. It writes:

- `figures/chapter-03-revenue-by-category.png`
- `figures/chapter-03-category-contributions.png`

## 8. A preview of statistics notation

Later chapters will develop formulas such as sample variance,

\[
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2,
\]

covariance,

\[
\operatorname{Cov}(X,Y)\propto\sum (x_i-\bar{x})(y_i-\bar{y}),
\]

and least squares,

\[
\sum_{i=1}^{n}(y_i-\hat y_i)^2.
\]

We do not yet know all the pieces of these formulas, but the summation symbol itself should already be readable: **calculate something for each observation, then add those results.** No derivations are needed yet.

## Mastery checkpoints

### Concept checkpoint

1. What does $i$ represent? What does $n$ represent?
2. Expand $\sum_{i=1}^{4}x_i$. What instruction does $\Sigma$ give?
3. Write $a_1+a_2+a_3$ with sigma notation.
4. Why can a quantity-weighted price differ from a simple average price?

### Hand calculation checkpoint

Use this table without code first:

| Row | Quantity | Price |
|---:|---:|---:|
| 1 | 1 | $18 |
| 2 | 4 | $6 |
| 3 | 5 | $3 |

Calculate revenue for every row, total revenue, total quantity, simple average price, and quantity-weighted average price. Then explain what each average treats equally. (Check: revenue totals $57 and quantity totals 10; finish both averages yourself.)

### Execution checkpoint

1. Group revenue by `day` instead of `category`.
2. Calculate day shares and confirm they sum to approximately 100%.
3. Pick one category and manually add its row revenues to verify pandas.
4. Propose a new transaction. Predict the direction of total revenue, transaction mean, total quantity, weighted price, and category share changes **before** rerunning.

### Interpretation checkpoint

**Category A has the highest average transaction value while Category B contributes the most total revenue. Is this contradictory?** No. A can have fewer, larger transactions while B has enough transactions to generate a larger total.

**A manager averages the average price of three categories and calls it the company's average selling price. What might be ignored?** Different categories may sell different quantities. Giving category means equal weight can misrepresent the average unit sold; the relevant quantities should weight prices.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** Summation appears throughout probability, expectation, variance, covariance, estimators, ANOVA, and regression.
- **BUAD 512B — Business Modeling with Python:** NumPy and pandas turn mathematical aggregations into efficient operations over analytical datasets.
- **BUAD 5112 — Competing Through Business Analytics:** Business metrics depend on exactly what is counted, summed, averaged, weighted, and compared.

This independent preparation chapter does not reproduce any William & Mary course.
