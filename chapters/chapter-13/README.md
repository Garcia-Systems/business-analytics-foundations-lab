# Chapter 13 — Seeing Data

> **Central question:** How can a picture reveal structure in data that is difficult to see in a table of numbers?

A restaurant group has 84 clean observations: four locations on each of 21 dates. One row is one **location-day**. The analysis-ready fields include date, location, revenue, transactions, customers, labor, weekday, and promotion status.

> analytical question → choose variables → choose visual form → inspect pattern → compare groups → look for anomalies → challenge the picture → interpret cautiously

Visualization is not decoration. It is a method for investigating data, comparing evidence, finding structure, and revealing where further analysis is needed.

## Learning objectives

By the end, you can choose plots from analytical questions; distinguish distribution, comparison, relationship, and time views; create and interpret histograms, boxplots, bar charts, scatterplots, and line plots; recognize skew, overlap, overplotting, and potential outliers; distinguish raw observations from aggregates; challenge axis and aggregation choices; and produce labeled, reproducible Matplotlib figures. You will also distinguish visual association from causal evidence.

## Start with the question, not the chart

Bad workflow: “I want to make a bar chart. What can I put in it?” Better workflow: “I want to know whether locations differ in typical daily revenue. What visual comparison would help answer that?” **The analytical question should determine the visual form.**

Four useful question types organize this chapter:

| question type | analytical question | useful form |
|---|---|---|
| Distribution | What values occur, and how common are they? | histogram or boxplot |
| Comparison | How do categories or groups differ? | aggregate bar chart or side-by-side boxplots |
| Relationship | How do two quantitative variables move together? | scatterplot |
| Time | How does a measure change over ordered time? | line plot |

These are a working vocabulary, not a chart encyclopedia.

## Tables and pictures are complementary

Consider 20 revenues: `1105, 1237, 1001, 1324, 1190, 1460, 985, 1282, 1154, 1398, 1078, 1215, 2134, 1170, 1342, 1044, 1261, 1435, 1128, 1308`. From the list alone, identify the typical value, spread, shape, and unusual observations. A histogram exposes concentration and the long upper tail more quickly—but the table identifies the exact unusual day, and numerical summaries precisely report count, mean, median, minimum, and maximum.

> Tables, numerical summaries, and plots answer complementary questions.

## Distribution: histogram and boxplot

```python
fig, ax = plt.subplots()
ax.hist(df["revenue"], bins=12)
ax.set(title="How Is Daily Revenue Distributed?",
       xlabel="Daily revenue ($)", ylabel="Location-days (count)")
```

The horizontal axis contains quantitative values, the vertical axis counts observations, and bins group values into ranges. Try `bins=6` and `bins=18`: apparent gaps or peaks can change. **A histogram is a summary, not a photograph of the data.** Inspect its center, spread, symmetry or skew, tails, modes, gaps, and unusual observations without yet calculating formal distribution-shape statistics.

```python
mean_revenue = df["revenue"].mean()
median_revenue = df["revenue"].median()
ax.axvline(mean_revenue, label="mean")
ax.axvline(median_revenue, linestyle="--", label="median")
```

A high upper tail can pull the mean above the median. Neither line replaces the distribution.

A boxplot compactly marks the median, the central portion from intuitive lower quartile \(Q_1\) to upper quartile \(Q_3\), whiskers, and potential outlier points. It connects to Chapter 11's 1.5-IQR candidate rule. A point beyond a whisker is a prompt to investigate, **not proof of an error**.

```python
names = sorted(df["location_name"].unique())
ax.boxplot([df.loc[df.location_name.eq(n), "revenue"] for n in names],
           tick_labels=names)
```

Side-by-side boxes reveal variability, skew, unusual days, and overlap hidden by means. If Location A has a higher mean, do all its days outperform Location B? **No.**

Promotion and non-promotion boxes answer a related comparison. Appropriate language is “Promotion days in this sample appear to have higher revenue,” not “promotions increased revenue.” Weekends, locations, expected demand, and promotion timing could explain the pattern.

## Comparison: bars represent aggregates

```python
summary = (df.groupby("location_name", as_index=False)
             .agg(revenue=("revenue", "sum"))
             .sort_values("revenue"))
ax.bar(summary["location_name"], summary["revenue"])
ax.set(title="Total Revenue by Location (Aggregate)",
       xlabel="Location", ylabel="Total revenue ($)")
```

Each bar is an aggregate, not one raw observation. Sorting by value can reveal structure more readily than arbitrary or alphabetical order, but ordering also directs attention.

A **histogram** bins one quantitative variable into numerical ranges. A **bar chart** gives values for categorical groups. Their gaps and bars may look similar, but the questions and meanings differ.

### Challenge the baseline

Two illustrative index values, A = 100 and B = 105, look dramatically different on a bar axis from 98 to 106. From zero, the five-unit difference appears in proportion. Ordinary bar lengths encode magnitude, so zero is normally important. This is not a blanket rule for every plot: a clearly labeled zoomed time series can reveal meaningful small changes.

Other choices deserve skepticism:

- too many categories make comparison unreadable;
- poor ordering can conceal structure;
- dual axes can manufacture persuasive alignment through arbitrary scales, so this chapter avoids them;
- averages hide distributions and overlap;
- missing observations do not announce themselves automatically;
- a cherry-picked time range can change the narrative.

> Charts are analytical constructions. They contain choices.

## Relationship: scatterplots and grain

```python
ax.scatter(df["labor_hours"], df["revenue"], alpha=0.4)
ax.set(title="Are Higher-Labor Days Associated with Higher Revenue?",
       xlabel="Labor hours", ylabel="Daily revenue ($)")
```

Look for direction, strength, shape, clusters, and unusual observations. Transparency makes dense regions visible; slight jitter can separate overlapping points when an axis is discrete. Color by location can answer whether groups cluster differently. Marker size by transactions might answer another question, but adding it merely for decoration increases cognitive load: **every visual encoding should answer a question.**

Ask, “What does one point represent?” In the plot above it is one location-day. After grouping to location means, one point is one location:

```python
location_means = df.groupby("location_name", as_index=False).agg(
    labor_hours=("labor_hours", "mean"), revenue=("revenue", "mean"))
```

Patterns can differ at different grains. **A graph inherits the grain of the data supplied to it.** Always label whether evidence is raw or aggregated.

### Observation is not explanation

- **Observation:** Higher-labor days tend to have higher revenue in this sample.
- **Possible explanation:** Managers may schedule more employees when they expect higher demand.
- **Unsupported conclusion:** Adding labor will increase revenue.

A visual relationship does not establish causation. Reverse direction, confounding, and group mix remain possible.

### Follow a visual anomaly back to its row

```python
q1, q3 = df["revenue"].quantile([0.25, 0.75])
iqr = q3 - q1
candidate = df.loc[df["revenue"] > q3 + 1.5 * iqr]
candidate[["date", "location_name", "revenue", "labor_hours", "promotion_active"]]
```

A plot says where to investigate; the table says what the observation actually is. Inspect transactions, customers, labor, date, and promotion context before choosing to retain, correct, or contextualize it.

## Time: order before connecting

A line is appropriate because dates have an ordered sequence:

```python
daily = (df.groupby("date", as_index=False)["revenue"].sum()
           .sort_values("date"))
ax.plot(daily["date"], daily["revenue"])
ax.set(title="Restaurant-Group Revenue over Time",
       xlabel="Date", ylabel="Daily revenue across locations ($)")
```

Weekday/weekend cycles and isolated spikes become visible; this is not time-series modeling. Calling `df.plot(x="date", y="revenue")` on unordered rows is unsafe. A line connects observations in row order. If data is not ordered meaningfully, the line can invent a visual story.

A log axis, when a positive metric spans orders of magnitude, changes visual spacing rather than observations:

```python
ax.set_yscale("log")
```

Equal vertical distances then mean multiplicative rather than equal additive change. It is not needed for this moderately ranged revenue dataset, which itself is an analytical choice.

## Reproducible exploratory experiment

```bash
python3 -m analytics_foundations chapter-13
```

The experiment loads deterministic location-day data, validates its composite key, prints numerical summaries, reconciles aggregates, retrieves IQR anomaly candidates, separates observation from causal conclusion, and creates eight labeled figures. Matplotlib uses a noninteractive backend; files have deterministic paths and figures close after saving.

This is primarily **exploratory visualization**, which helps an analyst discover what is happening. Explanatory visualization selects and communicates a finding to others; Chapter 34 will revisit that role. Production figures need descriptive titles, meaningful labels, and units. Neutral question titles are useful during exploration; finding titles require adequate evidence and must avoid unsupported causal language.

## Mastery checkpoints

### Concept checkpoint

Why is visualization part of analysis? What does a histogram answer? How does a bar chart differ? What does one scatterplot point represent? Why can an aggregate hide variation? Why can a truncated axis mislead? Why does association not prove causality? Why must line data be ordered?

### Chart-choice checkpoint

Choose and justify a form:

1. Distribution of transaction values? **Histogram or boxplot**, depending on whether detailed shape or compact summary matters.
2. Revenue by location? **Bar chart** for a specified aggregate; **boxplots** for distributions.
3. Is labor associated with revenue? **Scatterplot**, because both are quantitative.
4. How did revenue change over the month? **Line plot**, after sorting dates.

### Execution checkpoint

Create histograms with two bin choices; compare mean and median; create category boxplots; retrieve an unusual row; create a transparent scatterplot; group, sort, and plot location totals; aggregate and sort dates before a line; deliberately truncate a bar axis and explain the changed impression.

### Interpretation checkpoint

**Promotion days have higher average revenue. Did promotion cause it?** No. The association may reflect weekends, seasonality, location, expected demand, or promotion timing.

**One location has the highest total and several weak days. What next?** Examine transactions, labor, capacity, margins, weekday mix, variability, and promotion exposure.

## Part II: the workflow now available

> raw data → inspect → clean → transform → join → aggregate → visualize → question

Visualization is often where a new analytical question emerges: Why is one location more variable? Why do weekends differ? Are extreme days associated with promotions? Is labor responding to demand or driving it? These questions lead naturally into probability and statistics; Chapter 14 is not started here.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** distributions, variability, group differences, anomalies, and relationships should be understood visually before inference.
- **BUAD 512B — Business Modeling with Python:** pandas and Matplotlib support reproducible exploration of categorical and numerical business data.
- **BUAD 5112 — Competing Through Business Analytics:** visual business patterns are descriptive evidence, not automatically strategic or causal conclusions.

This independent preparation chapter does not reproduce any William & Mary course.

> **Central outcome:** Visualization turns structured data into visual evidence. Good plots help us see distributions, comparisons, relationships, time patterns, and anomalies, while careful analysts remain aware that every graph reflects choices about grain, aggregation, scale, and interpretation.
