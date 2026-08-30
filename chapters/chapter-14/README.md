# Chapter 14 — Events & Probability

> **Central question:** How can we describe uncertain business outcomes mathematically?

# Part III — Probability as a Model of Uncertainty

**Observed:** Yesterday 184 customers visited the restaurant. That is data. **Uncertain:** Tomorrow may be low, moderate, busy, or very busy. That outcome is not known.

> Descriptive analytics asks what happened in observed data. Probability asks what could occur and represents uncertainty before the result is known. Probability is useful when the outcome has not yet been observed.

Our path is:

> uncertain process → possible outcomes → sample space → event → probability → complement → union → intersection → simulation → business interpretation

## Random experiments, outcomes, and the sample space

A **random experiment** is a repeatable process whose exact outcome is uncertain before it occurs. Examples include tomorrow's demand category, whether a customer redeems an offer, whether an order arrives late, which category a shopper chooses, and whether a transaction is fraudulent. We need no philosophical theory of randomness to use this operational definition.

An **outcome** is one possible result. Tomorrow's demand outcome—sometimes denoted \(\omega\)—is one of `Low`, `Moderate`, `Busy`, and `Very Busy`. The **sample space** is the set of outcomes considered possible under the model:

\[
\Omega=\{L,M,B,V\}.
\]

The sample space is part of the model. If an analyst omits a possible outcome or defines outcomes incorrectly, the probability model is already wrong.

An **event** is a set of one or more outcomes from the sample space. Let

\[
A=\{\text{Busy},\text{Very Busy}\}.
\]

Event \(A\) means tomorrow is at least busy. `Busy` belongs to it (\(\text{Busy}\in A\)); `Low` does not (\(\text{Low}\notin A\)). \(\Omega\) denotes the whole sample space, and \(\varnothing\) is the empty set or impossible event.

## A valid probability model

| Demand state | Probability |
|---|---:|
| Low | 0.15 |
| Moderate | 0.35 |
| Busy | 0.30 |
| Very Busy | 0.20 |

The four states cannot occur together for the same day and cover all modeled possibilities, so

\[
0.15+0.35+0.30+0.20=1.
\]

For any event, \(P(A)\) means “the probability that event \(A\) occurs,” and

\[
0\le P(A)\le1,\qquad P(\Omega)=1,\qquad P(\varnothing)=0.
\]

A model such as `{Low: .15, Moderate: .35, Busy: .30, Very Busy: .30}` is invalid: its probabilities total 1.10. What is wrong? Mutually exclusive, exhaustive outcome probabilities must total 1, and each must be numeric and between 0 and 1.

\(P(A)=0.50\) can be said as 0.50, 50%, or about 1 in 2 **under the model**. It does not guarantee exactly half the outcomes in a small number of trials.

### Hand-worked events

1. Single event: \(P(B)=0.30\).
2. Grouped event: because demand states are mutually exclusive,
   \[P(B\cup V)=P(B)+P(V)=0.30+0.20=0.50.\]
3. Complement: \(A^c\) means that \(A\) does not occur. Here
   \[A^c=\{\text{Low},\text{Moderate}\},\qquad P(A^c)=1-P(A)=0.50.\]

## Equally likely outcomes are a special case

For a fair die, \(\Omega=\{1,2,3,4,5,6\}\). If \(A=\{2,4,6\}\), then

\[
P(A)=\frac{\text{number of favorable outcomes}}{\text{number of possible outcomes}}=\frac36=0.5.
\]

This shortcut works because the elementary outcomes are equally likely; it is **not** the universal definition of probability. Business outcomes usually are not equally likely. For example, if \(P(A)=.80,P(B)=.10,P(C)=.10\), event \(\{B,C\}\) contains two of three outcomes but has probability .20, not \(2/3\). Counting outcomes and adding their probability masses are different operations.

## OR, AND, and NOT

Use a fair promotional spin with outcomes 1 through 6. Let

\[
A=\{2,4,6\}\quad\text{(even)},\qquad B=\{4,5,6\}\quad\text{(at least 4)}.
\]

The **union** is OR, including either event or both:

\[
A\cup B=\{2,4,5,6\}.
\]

The **intersection** is AND:

\[
A\cap B=\{4,6\}.
\]

Because the overlapping outcomes would otherwise be counted twice, the addition rule is

\[
P(A\cup B)=P(A)+P(B)-P(A\cap B)=\frac36+\frac36-\frac26=\frac46.
\]

Two events are **mutually exclusive** if they cannot occur on one trial: \(A\cap B=\varnothing\). `Low` and `Very Busy` are mutually exclusive for one day, so their intersection has probability zero and their union probability is the sum. Mutually exclusive is not the same as independent; independence will be studied later.

Events are **exhaustive** if together they cover the whole sample space. The mutually exclusive categories satisfy

\[
L\cup M\cup B\cup V=\Omega,
\qquad P(L)+P(M)+P(B)+P(V)=1.
\]

The generated event, complement, union, and intersection diagrams show these set operations inside a rectangle representing \(\Omega\). Remember: OR includes overlap; AND is only overlap; NOT is everything outside the event.

## Where probabilities come from

Practical probabilities can come from:

1. **Symmetry or a known mechanism**, such as a fair die.
2. **Historical relative frequency**, such as 8% of observed deliveries being late.
3. **Judgment or model assumptions**, such as management assigning 20% to unusually high demand.

A probability can be mathematically valid and still be a poor business assumption. Chapter 13 might report that 20% of observed days were very busy; Chapter 14 asks whether we will model \(P(\text{Very Busy})=.20\) for a future day. That step assumes the history is relevant to the future process.

Given \(n\) observations, the **empirical probability** is

\[
\hat P(A)=\frac{\text{observed occurrences of }A}{n}.
\]

For 17 busy days among 50, \(\hat P(\text{Busy})=17/50=.34\). The hat lightly signals an estimate from observed data; it need not equal the unknown underlying probability.

## Probability in Python

The chapter keeps the representation transparent: a dictionary maps outcomes to probabilities, and an event is a set.

```python
from analytics_foundations.chapter_14 import (
    probability_of_event, complement_probability,
    simulate_categorical_outcomes, validate_probability_model,
)

model = {"Low": .15, "Moderate": .35, "Busy": .30, "Very Busy": .20}
event = {"Busy", "Very Busy"}
validate_probability_model(model)
probability_of_event(model, event)       # 0.50
complement_probability(model, event)     # 0.50
```

Validation rejects nonnumeric values, values outside \([0,1]\), totals not approximately 1 (with floating-point tolerance), and event outcomes outside \(\Omega\). It is deliberately a small educational helper, not an abstract probability framework.

## Make uncertainty executable with simulation

```python
import numpy as np

rng = np.random.default_rng(14)
states = np.array(["Low", "Moderate", "Busy", "Very Busy"])
probabilities = np.array([.15, .35, .30, .20])
simulated = rng.choice(states, size=1000, p=probabilities)
np.mean(np.isin(simulated, ["Busy", "Very Busy"]))
```

The simulation is random under the model, but a fixed seed makes the experiment reproducible. A seed does not make the conceptual process non-random; it lets another analyst reproduce this sequence.

Run 10, 100, 1,000, and 10,000 trials and compare their empirical busy-or-higher proportions with theoretical \(P(A)=.50\). Larger samples often settle nearer the modeled value, but convergence is not monotonic and no small batch is guaranteed. Repeating ten trials with different fixed seeds can produce results such as 30%, 60%, or 70%. Probability describes the process, not a guarantee about a small batch.

The cumulative simulation figure plots

```python
indicator = np.isin(simulated, ["Busy", "Very Busy"])
cumulative = np.cumsum(indicator) / np.arange(1, len(indicator) + 1)
```

against trial count, with .50 as a horizontal reference. This previews long-run stabilization without proving a law of large numbers.

> Simulation reproduces the probability model supplied to it. It does not automatically discover the true probability. A wrong input model is simulated faithfully.

## Models simplify; probability is not certainty

Is tomorrow really one of only four states? The category boundaries are choices:

- Low: fewer than 100 customers
- Moderate: 100–149
- Busy: 150–199
- Very Busy: 200+

Changing boundaries changes events and probabilities. “What is the probability tomorrow is busy?” is vague until busy is defined. A precise event is

\[
A=\{\text{customer count}\ge150\}.
\]

Define events before calculating. Ordinary words—unlikely, possible, likely, almost certain—are ambiguous. If two managers say “likely,” must they mean the same number? No; numerical probabilities expose the difference.

\(P(A)=.95\) does not make \(A\) certain. \(P(A)=.05\) does not make \(A\) impossible. Probability quantifies uncertainty; it does not eliminate it.

If \(P(\text{Busy or Very Busy})=.50\), that number informs but does not determine staffing. A choice also depends on overstaffing and understaffing costs, service standards, uncertainty in the estimate, and outcome consequences. Probability informs decisions; it does not make the decision itself. Expected-value optimization waits until Chapter 18.

## Reproducible experiment and visualizations

```bash
python3 -m analytics_foundations chapter-14
```

The command validates and prints the demand table; calculates a single event, complement, and mutually exclusive union; demonstrates the overlapping die union and intersection; simulates fixed-seed trial counts; distinguishes theoretical from empirical results; and writes:

- `figures/chapter-14-demand-probabilities.png` — probability distribution;
- `figures/chapter-14-event-in-sample-space.png` — Busy and Very Busy highlighted;
- `figures/chapter-14-complement.png` — event NOT occurring;
- `figures/chapter-14-union.png` — OR;
- `figures/chapter-14-intersection.png` — AND;
- `figures/chapter-14-simulation-convergence.png` — cumulative empirical proportion and .50 model reference.

## Common misconceptions

1. **“Probability .7 means exactly 7 occurrences in every 10 trials.”** It is a process model, not a guarantee for each block of ten.
2. **“More possible outcomes means more probability.”** Only under equal-likelihood assumptions.
3. **“Mutually exclusive means independent.”** They are different concepts; independence comes later.
4. **“Simulation automatically discovers truth.”** Simulation executes supplied assumptions; incorrect probabilities yield a faithful simulation of an incorrect model.

## Mastery checkpoints

### Concept checkpoint

Explain: What is a random experiment? An outcome? A sample space? An event? What does \(P(A)\) mean? Why do exhaustive mutually exclusive outcome probabilities sum to 1? What is a complement? How do union and intersection differ? What does mutually exclusive mean? Why is favorable divided by possible not always valid?

### Set checkpoint

Given \(\Omega=\{1,2,3,4,5,6\}\), \(A=\{2,4,6\}\), and \(B=\{4,5,6\}\), find \(A^c\), \(A\cup B\), \(A\cap B\), and their probabilities for a fair die. Check: \(A^c=\{1,3,5\}\), union \(=\{2,4,5,6\}\), intersection \(=\{4,6\}\), with probabilities \(3/6,4/6,2/6\).

### Execution checkpoint

Define and validate a categorical model; define a multi-outcome event; calculate its theoretical probability and complement; simulate 100, 1,000, and 10,000 trials; compare empirical with theoretical probabilities; then change the RNG seed and observe a different sequence.

### Interpretation checkpoint

- A restaurant assigns 70% to a busy night, but the night is slow. Was the model necessarily wrong? **No.** The 30% alternative was possible; one realization cannot validate or invalidate the estimate.
- In 20 simulations, 65% are busy-or-higher though the model says 50%. Is the code necessarily wrong? **No.** Small samples vary.
- An event has 3 of 5 possible outcomes. Must its probability be 60%? **Only if the five elementary outcomes are equally likely.**

## Preparation connections

- **BUAD 512A — Probability & Statistics with R:** probability is language for later random variables, distributions, sampling, estimation, and testing.
- **BUAD 512B — Business Modeling with Python:** simulation turns probability models into executable experiments.
- **BUAD 5112 — Competing Through Business Analytics:** decisions precede outcomes; probability quantifies uncertainty and exposes assumptions.

This independent preparation chapter does not reproduce any William & Mary course.

## What comes next—not yet

This chapter cannot fully answer, “What is the probability of a busy night **given** that it is Friday?” We can write the future question as

\[
P(\text{Busy}\mid\text{Friday}),
\]

but do not calculate it here. Chapter 15 will study how probabilities change when information is known.

> **Central outcome:** Probability gives us a mathematical language for uncertain outcomes. By defining a sample space, events, and probabilities, we can reason about what may happen before observing the result and use simulation to make that uncertainty executable.
