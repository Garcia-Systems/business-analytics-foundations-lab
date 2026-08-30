# Chapter 9 — Arrays & Vectorized Thinking

> **Central question:** How do we work with many numerical values efficiently and clearly in Python?

NumPy arrays turn mathematical relationships into operations over entire collections of data. Vectorization, aggregation, masking, and broadcasting let analytical code express **what** should happen to the data rather than repeatedly describing how to process one value at a time.

## Learning objectives

By the end, you can create arrays; interpret `shape`, `ndim`, `size`, and `dtype`; slice 1D and 2D arrays; perform element-wise arithmetic; explain vectorization; filter with Boolean masks; aggregate along axes; broadcast compatible shapes; reshape 1D values; distinguish slices from copies; locate extrema; and translate mathematical relationships into derived business metrics.

## The restaurant question

Five fictional restaurants are observed Monday through Sunday. Rows are locations, columns are days, and each entry is daily revenue:

```python
revenue = np.array([
    [4200, 4600, 4400, 5100, 6200, 7000, 5900],
    [3800, 4000, 4150, 4700, 5200, 6100, 5400],
    [4500, 4800, 5000, 5400, 6500, 7200, 6300],
    [3500, 3900, 4100, 4600, 5700, 6400, 5200],
    [4100, 4300, 4550, 5000, 6000, 6800, 5800],
])
```

Customer and labor-hour arrays have the same layout. We will ask which locations and days were strongest, what each restaurant earned per customer and labor hour, and which observations beat a target—without writing a loop for each cell.

## From a sequence to an analytical array

```python
revenues = [100, 120, 90]
revenues * 2                 # [100, 120, 90, 100, 120, 90]

revenues = np.array([100, 120, 90])
revenues * 2                 # array([200, 240, 180])
```

Python list multiplication repeats a sequence. NumPy defines arithmetic element by element. That semantic difference—not merely speed—is why an array naturally represents a numerical vector.

Useful constructors should answer a question rather than form a catalog:

```python
np.array([4200, 4600, 4400])  # observed revenue
np.arange(1, 8)                # day numbers 1 through 7
np.linspace(0, 1, 5)           # five evenly spaced scenario weights
np.zeros(7)                     # initialize seven daily adjustments
np.ones(7)                      # seven multiplicative baseline factors
```

## Shape, dimension, size, and dtype

```python
x = np.array([10, 20, 30])
x.shape    # (3,)
x.ndim     # 1
x.size     # 3

X = np.array([[10, 20, 30], [40, 50, 60]])
X.shape    # (2, 3)
X.ndim     # 2
X.size     # 6
```

`X` is the computational form of the Chapter 7 matrix $X\in\mathbb{R}^{2\times3}$. Shape reports the length of each dimension; dimensionality reports the number of axes; size counts every entry.

An array normally has one consistent storage type:

```python
np.array([1, 2, 3]).dtype       # an integer dtype
np.array([1.0, 2.0, 3.0]).dtype # a floating dtype
np.array([1, 2.5, 3]).dtype     # floating: integers are promoted
```

Exact dtype widths can vary by platform. Practically, division and missing values commonly require floating-point representation.

## Indexing and analytical slicing

For `matrix[i, j]`, row `i` and column `j` identify the entry $x_{ij}$ (Python positions start at zero):

```python
revenue[0]           # one complete location row, shape (7,)
revenue[1:4]         # locations 1, 2, 3, shape (3, 7)
revenue[0, :]        # all days for location 0, shape (7,)
revenue[:, 2]        # Wednesday for every location, shape (5,)
revenue[1:3, 2:5]    # two locations by three days, shape (2, 3)
```

Predict the shape before executing. Selecting one integer position removes that dimension; selecting a slice preserves it.

## Element-wise relationships and vectorization

Revenue, cost, customer, price, quantity, and labor arrays allow direct relationships:

```python
profit = revenue - cost
revenue_per_customer = revenue / customers
labor_cost = labor_hours * hourly_wage
sales = price * quantity
```

If $p_i=q_i r_i$ for every observation $i$, NumPy lets us state the relationship once. Compare:

```python
profit = []
for r, c in zip(revenue, cost):
    profit.append(r - c)

profit = revenue - cost
```

Loops remain useful when work is genuinely sequential or irregular. For array-wide numerical relationships, the vectorized form is usually closer to the mathematics, shorter, easier to inspect, and efficient on large arrays. **Expressiveness comes first; performance is a benefit.** In the restaurant matrix:

```python
revenue_per_customer = revenue / customers       # (5, 7) / (5, 7) → (5, 7)
revenue_per_labor_hour = revenue / labor_hours    # (5, 7) / (5, 7) → (5, 7)
```

Every output cell retains the matching location/day meaning. A high total and high efficiency are different claims, so always interpret the denominator.

## Comparisons, masks, and conditional transformation

Comparisons are vectorized too:

```python
revenue > 5000                  # Boolean array, shape (5, 7)
high_revenue = revenue[revenue > 5000]  # matched values, flattened to 1D
mask = (revenue > 5000) & (labor_hours < 100)
revenue[mask]
```

Use parentheses around each comparison and `&` (and), `|` (or), and `~` (not). Python's `and` and `or` ask for one truth value from each whole object; an array contains many truth values, so its truth is ambiguous.

`np.where` conditionally transforms every cell while preserving shape:

```python
performance = np.where(
    revenue >= targets[:, np.newaxis],
    "above_target",
    "below_target",
)
```

## Aggregation and the axis that disappears

NumPy's `sum`, `mean`, `min`, `max`, and `median` make summation and statistical notation executable. For example, `np.mean(x)` computes \(\bar{x}=\frac{1}{n}\sum_i x_i\).

For $R\in\mathbb{R}^{5\times7}$:

```python
revenue.sum()          # scalar: all 35 entries
revenue.sum(axis=0)    # (5, 7) → (7,), one total per day
revenue.sum(axis=1)    # (5, 7) → (5,), one total per location
revenue.mean(axis=1)   # (5, 7) → (5,), one daily average per location
```

Do not memorize axis directions without meaning. `axis=0` removes the row dimension by aggregating **down rows**, leaving columns/days. `axis=1` removes the column dimension by aggregating **across columns**, leaving rows/locations. Reason about which dimension disappears.

```python
by_location = revenue.sum(axis=1)
best_location = np.argmax(by_location)
worst_location = np.argmin(by_location)
```

`argmax` and `argmin` return positions, not values. Use the position to index the location label or total. The same procedure on `revenue.sum(axis=0)` finds the strongest and weakest day.

## Broadcasting: one target per location

Each location has one daily target:

```python
targets = np.array([5000, 4800, 5200, 4500, 5100]) # (5,)
target_column = targets[:, np.newaxis]               # (5, 1)
deviation = revenue - target_column                  # (5, 7)
```

Shape reasoning makes the intention visible:

```text
revenue                 (5, 7)
targets[:, np.newaxis]  (5, 1)
--------------------------------
difference              (5, 7)
```

NumPy conceptually stretches the size-1 day dimension across seven compatible columns without first physically copying target data. Comparing shapes from the right, dimensions are compatible when they are equal or either is 1. Thus `(5, 7)` and `(5, 1)` work. `(5, 7)` and `(5,)` do **not**: trailing dimensions compare as 7 and 5. A `(7,)` vector would broadcast across days, but that would encode a different business meaning.

## Reshaping and 1D ambiguity

```python
x = np.array([10, 20, 30])
x.shape                 # (3,): one axis, neither row nor column
x.reshape(3, 1).shape   # (3, 1): explicit column
x[:, np.newaxis].shape  # (3, 1): another explicit column
```

A 1D NumPy array has no distinct row/column orientation. Chapters 6 and 7 used geometric vectors and matrices; numerical matrix work sometimes needs a reshape to make the intended orientation and broadcasting explicit. Reshaping changes organization, not the number of values.

## Universal functions and a scale preview

Universal functions apply to each entry:

```python
np.sqrt(revenue)
np.log(revenue)
np.exp(np.log(revenue))
np.abs(deviation)
```

Dozens exist, but the central pattern matters more than a catalog. A preview of statistical standardization makes it especially clear:

```python
z = (x - x.mean()) / x.std(ddof=1)
```

This implements $z_i=(x_i-\bar{x})/s$: array → subtract scalar → divide by scalar → transformed array. It does not establish that standardization is appropriate; it only executes the chosen transformation.

## Views, copies, and missing-value preview

A slice can be a **view** into the original storage:

```python
x = np.array([10, 20, 30, 40, 50])
subset = x[1:4]
subset[0] = 999       # may also change x

subset = x[1:4].copy()
subset[0] = 999       # x remains unchanged
```

Be deliberate when modifying slices. We need no deep memory-layout theory yet.

NumPy represents a numerical missing value with `np.nan`; `np.mean([1, np.nan])` is `nan`, while `np.nanmean([1, np.nan])` ignores it. This is only a preview—Chapter 11 will address messy data and whether ignoring values is defensible.

## Executable experiment and figures

Run:

```bash
python3 -m analytics_foundations chapter-09
```

The experiment reports shapes and dtypes; contrasts lists and arrays; slices the matrix; computes revenue/customer and revenue/labor hour; filters a two-condition mask; reconciles total and axis aggregations; locates best/worst positions; broadcasts location targets; reshapes a vector; standardizes location totals; and generates:

1. a Matplotlib revenue-matrix image with labeled rows, columns, and cells;
2. a location-total bar chart built from `revenue.sum(axis=1)`, a dimension-reducing aggregation;
3. a target-deviation chart built directly from the `(5, 7) - (5, 1)` broadcast.

## Mastery checkpoints

### Concept checkpoint

1. How is a NumPy array different from a Python list?
2. What do `shape` and `ndim` mean?
3. What is vectorization, and why is expressiveness the first benefit?
4. What is a Boolean mask? Why use `&` rather than `and`?
5. What does aggregation along an axis do?
6. Why can broadcasting stretch a dimension of size 1?
7. What is the difference between `(3,)` and `(3, 1)`?

### Hand-reasoning checkpoint

Given `X.shape == (4, 6)`, predict before executing:

```python
X.sum()          # scalar, shape ()
X.sum(axis=0)   # (6,): row dimension disappears
X.sum(axis=1)   # (4,): column dimension disappears
X[:, 2]         # (4,): integer column selection removes that dimension
X[1:3, :]       # (2, 6): slices preserve dimensions
```

Cover the comments, reason from the indexing or disappearing dimension, and then verify in Python.

### Execution checkpoint

Add a sixth restaurant row to revenue, customers, and labor hours. Predict the new shapes. Calculate revenue per customer, create a mask combining two conditions, and locate its observations. Calculate both axis totals and reconcile either sum with the overall total. Define six location targets, reshape them into a column, broadcast target deviations, and confirm the output shape.

### Interpretation checkpoint

**Location A has the largest total revenue. Is it automatically most efficient?** No. Total is sensitive to scale. Compare revenue/customer, revenue/labor hour, and margin, while checking whether their denominators and time periods support the decision.

**A vectorized expression returns correct numbers. Is the metric necessarily meaningful?** No. Computational correctness cannot validate the metric definition, denominator, target, data quality, causal interpretation, or decision relevance.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** statistical formulas operate over observations; vectorized numerical thinking supports later distributions, estimation, and regression.
- **BUAD 512B — Business Modeling with Python:** NumPy arrays, slicing, broadcasting, and matrix-style calculations support later wrangling, visualization, modeling, and numerical work.
- **BUAD 5112 — Competing Through Business Analytics:** fast calculation is useful only when the metric, denominator, comparison, and interpretation are valid.

This independent preparation chapter does not reproduce any William & Mary course.
