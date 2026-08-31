# Chapter 15 — Conditional Probability

> **Central question:** How should our probability assessment change when we learn something relevant?

Chapter 14 left one question unresolved: **What is the probability of a busy night given that it is Friday?** Write it

\[
P(B\mid F).
\]

The vertical bar means **“given that.”** Thus (P(B\mid F)) is the probability of event (B), given that event (F) is known to have occurred.

Our path is:

> prior event probability → new information → restricted sample space → conditional probability → contingency table → multiplication rule → independence → Bayes' rule → business interpretation

## Begin with the restricted sample space, not a formula

The restaurant's 100 observed days are classified as follows:

| | Busy | Not Busy | Total |
|---|---:|---:|---:|
| Friday | 18 | 6 | 24 |
| Not Friday | 22 | 54 | 76 |
| Total | 40 | 60 | 100 |

Without information about the day, all 100 observations are relevant, so

\[
P(\text{Busy})=\frac{40}{100}=0.40.
\]

Once Friday is known, only the Friday row remains relevant: 24 observations, of which 18 were busy. Therefore

\[
P(B\mid F)=\frac{18}{24}=0.75.
\]

> **Conditioning changes the denominator.** New information restricts the set of outcomes under consideration.

Only after seeing that intuition do we write the general formula:

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.
\]

Here (A\cap B) means both events occur. Dividing by (P(B)) rescales the part where both occurred relative to the restricted region where (B) occurred. In the restaurant data,

\[
P(\text{Busy}\cap\text{Friday})=\frac{18}{100}=0.18,
\qquad P(\text{Friday})=\frac{24}{100}=0.24,
\]

and hence

\[
P(\text{Busy}\mid\text{Friday})=\frac{0.18}{0.24}=0.75.
\]

## Reverse conditioning changes the question

(P(\text{Busy}\mid\text{Friday})) is not (P(\text{Friday}\mid\text{Busy})). Reversing the bar changes the restricted group:

\[
P(\text{Busy}\mid\text{Friday})=\frac{18}{\color{#b44}{24}}=0.75,
\qquad
P(\text{Friday}\mid\text{Busy})=\frac{18}{\color{#b44}{40}}=0.45.
\]

- **given Friday** → denominator 24;
- **given Busy** → denominator 40.

The same numerator appears here, but the questions and denominators differ.

## Contingency tables: joint, marginal, and conditional

A **contingency table** cross-classifies observations by two categorical variables. Interior cells are combinations, the outer cells are **row totals** and **column totals**, and the bottom-right value is the **grand total**.

- A **joint probability**, (P(A\cap B)), is the probability that both occur: (P(B\cap F)=18/100=.18).
- A **marginal probability** comes from a row or column total: (P(B)=40/100=.40) and (P(F)=24/100=.24).
- A **conditional probability** is calculated within a restricted row or column: (P(B\mid F)=18/24=.75).

Dividing every count by 100 produces the joint probability table:

| | Busy | Not Busy | Total |
|---|---:|---:|---:|
| Friday | 0.18 | 0.06 | 0.24 |
| Not Friday | 0.22 | 0.54 | 0.76 |
| Total | 0.40 | 0.60 | 1.00 |

The count and probability tables express identical relationships on different scales.

## The multiplication rule and probability tree

Rearranging the conditional formula gives

\[
P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A).
\]

For Friday and Busy,

\[
P(B\cap F)=P(B\mid F)P(F)=0.75(0.24)=0.18.
\]

Read a tree path sequentially: the probability of reaching Friday, then the probability of Busy inside Friday. Multiply probabilities along the path to get its joint leaf probability. The generated tree also shows the Not Friday branch and all four joint leaves.

## Independence—and what it is not

Events (A) and (B) are **independent** when learning that one occurred does not change the probability of the other. Equivalent conditions are

\[
P(A\mid B)=P(A)\quad(P(B)>0),
\qquad P(A\cap B)=P(A)P(B).
\]

Here (P(Busy)=.40), while (P(Busy\mid Friday)=.75). They differ, so Busy and Friday are **not independent in these data/model**.

A brief independent example uses separate fair experiments. A coin result does not alter a die result:

\[
P(H\cap6)=P(H)P(6)=\frac12\frac16=\frac1{12}.
\]

Do not confuse independence with **mutual exclusivity**:

\[
\text{mutually exclusive: }P(A\cap B)=0,
\qquad
\text{independent: }P(A\cap B)=P(A)P(B).
\]

For one day, `Low demand` and `Very Busy demand` cannot both happen. If both have nonzero probability, their product is nonzero but their intersection is zero. Knowing Low occurred makes Very Busy impossible, so these mutually exclusive events are dependent. In general, two nonzero-probability mutually exclusive events cannot be independent.

## Make the denominator explicit in pandas

The chapter dataset has one row per observed day, with `date`, `is_friday`, `busy`, `promotion_active`, and `rain`. Calculate the condition manually first:

```python
from analytics_foundations.chapter_15 import build_day_dataset

df = build_day_dataset()
friday_rows = df[df["is_friday"]]
p_busy_given_friday = friday_rows["busy"].mean()
```

Because Boolean `True` is represented as 1, the mean is the proportion:

\[
\frac{\#(Busy\cap Friday)}{\#(Friday)}=\frac{18}{24}.
\]

Then pandas can provide counts and row-conditional proportions:

```python
table = pd.crosstab(df["is_friday"], df["busy"])
row_proportions = pd.crosstab(
    df["is_friday"],
    df["busy"],
    normalize="index",
)
```

`normalize="index"` makes each row sum to one, so it conditions on the row variable. Syntax supports the reasoning; it does not replace deciding which denominator answers the question.

### Multiple conditions and sparse evidence

We can ask

\[
P(Busy\mid Friday\cap Promotion),
\]

the probability of Busy given both Friday **and** an active promotion. A Boolean mask can restrict to both. In this constructed dataset, only three observations meet that condition and two are busy. The estimate (2/3) may look large, but it rests on very little evidence.

> More specific conditioning gives more relevant information but often leaves fewer observations.

This is a sample-size warning, not yet a confidence-interval analysis.

## Bayes by counts first: fraud alerts

Direction matters dramatically in a rare-event business setting. Suppose

\[
P(Fraud)=.01,\quad P(Alert\mid Fraud)=.90,
\quad P(Alert\mid No\ Fraud)=.05.
\]

The system catches 90% of fraud, but the business question after an alert is (P(Fraud\mid Alert))—**not** (P(Alert\mid Fraud)).

Start with 10,000 transactions:

| Actual status | Alert | No alert | Total |
|---|---:|---:|---:|
| Fraud | 90 | 10 | 100 |
| No fraud | 495 | 9,405 | 9,900 |
| Total | 585 | 9,415 | 10,000 |

The expected counts follow directly: (10{,}000(.01)=100) fraud transactions, (100(.90)=90) true alerts, and (9{,}900(.05)=495) false alerts. Thus

\[
P(Fraud\mid Alert)=\frac{90}{90+495}=\frac{90}{585}\approx0.154.
\]

Only about 15.4% of alerts correspond to fraud under these assumptions. The 9,900 non-fraud transactions create many false alerts even though their false-positive rate is only 5%.

## Bayes' rule reverses conditioning

Bayes' rule is

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
\]

For a binary target, expand the denominator into the two ways (B) can occur:

\[
P(B)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c),
\]

so

\[
P(A\mid B)=
\frac{P(B\mid A)P(A)}
{P(B\mid A)P(A)+P(B\mid A^c)P(A^c)}.
\]

For fraud,

\[
\frac{.90(.01)}{.90(.01)+.05(.99)}
=\frac{.009}{.0585}\approx.1538,
\]

which reconciles exactly with (90/585).

- **Prior:** (P(Fraud)=.01), before seeing an alert.
- **Likelihood-like evidence:** (P(Alert\mid Fraud)=.90). (Formal likelihood theory is beyond this chapter.)
- **Posterior:** (P(Fraud\mid Alert)\approx.154), after observing an alert.

Bayes reverses the direction of conditioning. A highly sensitive alert can still have a modest posterior when its target has a low **base rate**. Ignoring that base rate is **base-rate neglect**. The same issue appears in equipment-failure alarms, churn alerts, defect inspection, and screening systems.

## Transparent Python and conditional simulation

The module deliberately offers small helpers rather than a symbolic probability package:

```python
from analytics_foundations.chapter_15 import (
    bayes_binary, conditional_probability, simulate_fraud_alerts,
)

conditional_probability(.18, .24)  # .75
bayes_binary(.01, .90, .05)        # about .154
```

Zero-probability conditioning raises a clear error. Simulation first generates fraud status and then generates an alert using the probability appropriate to each status:

```python
rng = np.random.default_rng(seed)
fraud = rng.random(n) < fraud_rate
alert = np.empty(n, dtype=bool)
alert[fraud] = rng.random(fraud.sum()) < sensitivity
alert[~fraud] = rng.random((~fraud).sum()) < false_positive_rate
fraud_given_alert = fraud[alert].mean()
```

A fixed seed makes the experiment reproducible. With many transactions, the empirical posterior should be reasonably near the theoretical one, though not identical.

> Simulation does not rescue bad assumptions. Incorrect fraud rates, sensitivities, or false-positive rates produce a faithful simulation of incorrect inputs.

## Seven hand-worked examples

1. Busy given Friday: (P(B\mid F)=18/24=.75).
2. Friday given Busy: (P(F\mid B)=18/40=.45).
3. Joint probability: (P(B\cap F)=18/100=.18).
4. Multiplication rule: (.75\times.24=.18).
5. Independence check: (P(B)=.40\ne P(B\mid F)=.75), so dependent here.
6. Bayes by counts: (90/(90+495)=.1538).
7. Bayes by formula: (.90(.01)/[.90(.01)+.05(.99)]=.1538).

## Business interpretation and causality

Knowing (P(Busy)=.40) but (P(Busy\mid Friday)=.75) is useful planning information. It does not automatically determine staffing. Managers must also consider labor cost, poor-service cost, expected revenue, uncertainty in the estimate, promotions, weather, and reservations.

The inequality

\[
P(Busy\mid Friday)>P(Busy)
\]

shows **association** in the model/data, not that Friday itself causally produces demand. Customer weekday behavior, Friday promotions or events, different hours, or seasonality could explain the relationship. Conditional probability alone does not identify a causal mechanism.

## Common misconceptions

1. **“(P(A\mid B)=P(B\mid A)).”** The conditioning event changes the denominator; the values can differ dramatically.
2. **“A 90% accurate/sensitive alert means 90% of alerts are true.”** The base rate and false-positive rate matter.
3. **“Mutually exclusive events are independent.”** Nonzero mutually exclusive events are dependent.
4. **“Conditional probability proves causality.”** It quantifies an association, not a causal mechanism.
5. **“A highly specific subgroup estimate must be better.”** It may contain too few observations to be stable.

## Reproducible experiment and visualizations

```bash
python3 -m analytics_foundations chapter-15
```

The command builds the day-level data, prints its grain and contingency table, reports marginal/joint/forward/reverse probabilities, verifies multiplication, evaluates independence, reconciles Bayes by counts and formula, runs a fixed-seed simulation, interprets the result, states limitations, and writes:

- `figures/chapter-15-restricted-denominator.png` — all 100 days versus the 24 relevant Fridays;
- `figures/chapter-15-contingency-table.png` — annotated Friday × Busy counts;
- `figures/chapter-15-probability-tree.png` — conditional branches and joint leaves;
- `figures/chapter-15-bayes-counts.png` — fraud, non-fraud, true alerts, and false alerts.

## Mastery checkpoints

### Concept checkpoint

What does (P(A\mid B)) mean? What changes when conditioning on (B)? What are joint and marginal probabilities? What is the multiplication rule? What does independence mean? How does it differ from mutual exclusivity? What does Bayes' rule accomplish?

### Denominator checkpoint

| | Purchase | No Purchase | Total |
|---|---:|---:|---:|
| Email opened | 30 | 70 | 100 |
| Email not opened | 20 | 180 | 200 |
| Total | 50 | 250 | 300 |

Write the denominator explicitly, then find (P(Purchase)), (P(Purchase\mid Opened)), (P(Opened\mid Purchase)), and (P(Purchase\cap Opened)).

Check: (50/300=1/6), (30/100=.30), (30/50=.60), and (30/300=.10). Notice how each question chooses its denominator.

### Independence checkpoint

Given (P(A)=.4), (P(B)=.5), and (P(A\cap B)=.2), are the events independent? Yes: (P(A)P(B)=.4(.5)=.2). If the intersection changes to .1, are they independent? No: (.1\ne.2).

### Execution checkpoint

Build a crosstab; calculate a conditional probability manually; reverse its direction; calculate a joint probability; verify multiplication; test independence; calculate a Bayes posterior; and simulate a binary screening process with a local NumPy generator.

### Interpretation checkpoint

- **75% of Friday nights are busy. Does that mean 75% of busy nights are Fridays?** No; (18/24\ne18/40).
- **A fraud model catches 90% of fraud. Does an alert imply 90% fraud probability?** No; the fraud base rate and false-positive rate matter.
- **Busy demand is more common on Fridays. Does that prove Friday causes it?** No; conditional probability describes association, not a causal mechanism.

## Preparation connections

- **BUAD 512A — Probability & Statistics with R:** conditional probability supports independence, joint distributions, Bayesian reasoning, sampling models, and later inference.
- **BUAD 512B — Business Modeling with Python:** contingency tables and simulation make formulas executable and checkable.
- **BUAD 5112 — Competing Through Business Analytics:** contextual probabilities can improve decisions when conditions and assumptions are meaningful.

This independent preparation chapter does not reproduce any William & Mary course.

> **Central outcome:** Conditional probability changes the denominator to reflect what is already known. This makes it possible to quantify context, test independence, reason through sequential events, and use Bayes' rule to reverse conditional relationships without confusing one probability for another.
