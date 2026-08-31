# Chapter 16 — Random Variables

> **Central question:** How can an uncertain business outcome become a numerical object that we can calculate with?

The path is **random experiment → outcome → numerical mapping → random variable → possible values → probability distribution → discrete versus continuous → probability calculations → simulation → business interpretation**.

> A random variable maps uncertain outcomes to numerical values. Its distribution tells us how probability is allocated across those values, allowing uncertainty to be described with the same mathematical and computational tools we use for ordinary numerical data.

Expected value waits until Chapter 18; named distributions wait until Chapter 17.

## From outcomes to numbers

Tomorrow's restaurant demand state might be Low, Moderate, Busy, or Very Busy. Define

\[
X(\text{Low})=80,\quad X(\text{Moderate})=120,\quad X(\text{Busy})=180,\quad X(\text{Very Busy})=240.
\]

**The random variable does not create randomness. The experiment is uncertain; the random variable assigns a numerical value to each possible outcome.** This repeats Chapter 1's function idea: \(X:\Omega\rightarrow\mathbb R\) maps each outcome to a real number.

| Outcome | \(X(\omega)\) | Probability |
|---|---:|---:|
| Low | 80 | 0.15 |
| Moderate | 120 | 0.35 |
| Busy | 180 | 0.30 |
| Very Busy | 240 | 0.20 |

Capital \(X\) denotes the uncertain quantity; lowercase \(x\) is a particular possible value. Thus \(X=160\) says the realized value is 160, while \(P(X=x)\) means the probability that \(X\) takes value \(x\).

## Customer-count PMF

Let \(X=\text{number of customers tomorrow}\):

| Customer count \(x\) | \(P(X=x)\) |
|---:|---:|
| 80 | 0.10 |
| 120 | 0.20 |
| 160 | 0.35 |
| 200 | 0.25 |
| 240 | 0.10 |

The **probability mass function (PMF)** is \(p_X(x)=P(X=x)\). It assigns mass to each possible numerical value. A valid PMF requires \(p_X(x)\geq0\) and \(\sum_xp_X(x)=1\). Here

\[
0.10+0.20+0.35+0.25+0.10=1.
\]

Hand-worked examples are:

\[
P(X=160)=0.35,
\]
\[
P(X\geq200)=P(X=200)+P(X=240)=0.25+0.10=0.35,
\]
\[
P(120\leq X\leq200)=0.20+0.35+0.25=0.80.
\]

Complements still work: \(P(X<200)=1-P(X\geq200)=0.65\). Thresholds turn numerical variables back into events: **random variable → condition → event → probability**. For comfortable capacity 180, \(P(X>180)=.35\); \(P(X\leq120)=.30\).

### Accumulation and the CDF

First accumulate: \(P(X\leq80)=.10\), \(P(X\leq120)=.30\), and \(P(X\leq160)=.65\). The **cumulative distribution function (CDF)** is \(F_X(x)=P(X\leq x)\). Hence

\[
F_X(160)=0.10+0.20+0.35=0.65.
\]

The PMF asks “how much probability sits exactly here?” The CDF asks “how much has accumulated up to here?”—a Chapter 5 connection. A discrete CDF jumps: it is 0 before 80, .10 at 80, .30 at 120, and eventually 1.

**Table experiment:** show the matching set and arithmetic for \(P(X=160)\), \(P(X\geq200)\), \(P(X<160)\), \(P(120\leq X\leq200)\), and \(P(X\ne160)\).

## Discrete versus continuous

A **discrete random variable** has countable/listable values. Customers, returns, defects, late deliveries, purchases, and support tickets are examples. Values need not be consecutive: \(X\in\{80,120,160,200,240\}\) is discrete.

A **continuous random variable** can conceptually take any value in an interval. Let \(Y=\text{order preparation time in minutes}\), with possibilities \(8.2,8.21,8.213,\ldots\). Processing time, delivery duration, weight, temperature, wait time, and continuously modeled revenue are examples.

| Question | Discrete | Continuous |
|---|---|---|
| Possible values | countable/listable | continuum |
| Example | customers | wait time |
| Exact-value probability | can be positive | 0 |
| Representation | mass | area/density |
| Treatment here | detailed | conceptual preview |

For continuous \(Y\), \(P(Y=10)=0\), even if a measurement is recorded as 10.0. Probability belongs to intervals such as \(P(9\leq Y\leq11)\), not infinitely precise points. Later,

\[
P(a\leq Y\leq b)=\int_a^b f_Y(y)\,dy,
\]

which explains why Chapter 5 integration matters. As a preview, \(F_Y(y)=P(Y\leq y)\) and \(P(a<Y\leq b)=F_Y(b)-F_Y(a)\). No density is fitted and no named distribution is assumed here.

## Model, realization, and dataset

Before tomorrow, \(X=\) tomorrow's customer count is uncertain. After 174 visit, \(X=174\) is the realized observation. A `daily_customers` column contains observations \(x_1,\ldots,x_n\); random variable \(X\) belongs to a probability model believed to generate values like them. **A dataset column is observed data. A random variable is part of a probability model.**

## Transformations and indicators

If revenue per customer is approximately $18, define \(R=18X\). When \(X=160\), \(R=2880\). A function of a random variable is itself random; this connects Chapter 1 and Chapter 9 without calculating expected revenue. A nonlinear capacity transform is \(W=\max(X-180,0)\).

Define an indicator

\[
I=\begin{cases}1,&X\geq200\\0,&X<200.\end{cases}
\]

Then \(P(I=1)=.35\). Python's `high_demand = customers >= 200` and `indicator = high_demand.astype(int)` implement it. Indicators later connect probability, averages, regression, and classification. A 0/1 variable is often modeled with a Bernoulli distribution, but Chapter 17 handles named distributions.

Two variables can describe a day: \(X=\) customers and \(Y=\) labor hours produce a pair \((X,Y)\). Joint distributions wait; this only prepares for Chapter 19.

## Python and simulation

```python
values = np.array([80, 120, 160, 200, 240])
probabilities = np.array([0.10, 0.20, 0.35, 0.25, 0.10])
probabilities.sum()
probabilities[values >= 200].sum()
```

Small `validate_pmf`, `exact_probability`, `threshold_probability`, `probability_between`, and `discrete_cdf` helpers keep the representation transparent. Simulation uses:

```python
rng = np.random.default_rng(seed)
samples = rng.choice(values, size=1000, p=probabilities)
unique, counts = np.unique(samples, return_counts=True)
```

Each draw is one realization. **The PMF belongs to the probability model; empirical proportions belong to a simulated or observed sample.** Finite proportions need not exactly match the model.

```bash
python3 -m analytics_foundations chapter-16
```

The command creates an outcome mapping, PMF, step CDF, model-versus-sample comparison, and explicitly conceptual continuous-area preview.

## Common misconceptions

1. **“Python creates the random variable.”** No: it is a mathematical mapping from uncertain outcomes to numbers.
2. **“\(X\) and \(x\) are identical.”** \(X\) is uncertain; \(x\) is possible or realized.
3. **“A continuous point has positive probability.”** Exact-point probability is zero under a continuous model.
4. **“A PMF is a histogram.”** A PMF specifies a model; a histogram summarizes data.
5. **“A pandas column is automatically a random variable.”** Calling observations realizations requires a modeling assumption.

## Mastery checkpoints

### Concept

What is a random variable and how is it a function? Distinguish \(X\) from \(x\), PMF from CDF, discrete from continuous, and a dataset column from a random variable. Why is \(P(Y=10)=0\)?

### Probability

Given \(P(X=0,1,2,3)=(.10,.25,.40,.25)\), calculate—with arithmetic—\(P(X=2)\), \(P(X\geq2)\), \(P(X<3)\), \(P(1\leq X\leq2)\), and \(F_X(1)\).

### Classification

Classify and explain: number of purchases, delivery time, number of returns, product weight, daily support tickets, and checkout duration.

### Execution

Define and validate a PMF; calculate an interval probability and CDF values; simulate 1,000 realizations; compare empirical and theoretical proportions; define an indicator; and transform the variable.

### Interpretation

* If \(P(X=200)=.25\) and one day has 200 customers, was the model “25% correct”? **No:** .25 was an ex-ante probability, not an after-the-fact score.
* If 29% of 100 simulations equal 160 while the PMF says 35%, is code necessarily wrong? **No:** finite samples vary.
* Can exactly 20.000000… minutes have positive probability under a continuous model? **No:** probability is assigned to intervals.

## W&M preparation connection

* **BUAD 512A — Probability & Statistics with R:** random variables underlie distributions, expectations, variance, sampling distributions, and inference.
* **BUAD 512B — Business Modeling with Python:** NumPy makes discrete models and simulations executable.
* **BUAD 5112 — Competing Through Business Analytics:** demand, delays, defects, revenue, and counts become useful measurable uncertainties.

These are preparation connections only; this independent chapter does not reproduce a William & Mary course.
