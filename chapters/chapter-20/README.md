# Chapter 20 — Monte Carlo Business

> **Central question:** When a business outcome depends on several uncertain quantities at once, how can simulation show the range of possible results before a decision?

James River Restaurant Group is considering a special Friday promotion at one location. The unit of analysis is one evening; thousands of simulated evenings reveal risk hidden by a point forecast.

> **Monte Carlo simulation** repeatedly generates plausible uncertain inputs from a probability model, evaluates the business outcome, saves it, and studies the resulting distribution. It converts explicit assumptions into possible business results—not predictions of the future.

The decision chain is: **business decision → uncertain inputs → probability models → dependence assumptions → simulated scenario → business outcome → repetition → outcome distribution → risk metrics → sensitivity → recommendation → limitations**.

## A transparent business model

\[
Revenue=Customers\times AverageSpend
\]
\[
FoodCost=0.30\times Revenue,\quad LaborCost=LaborHours\times\$18
\]
\[
PromotionCost=\$300+RedeemedOffers\times\$4
\]
\[
Profit=Revenue-FoodCost-LaborCost-PromotionCost
\]

| Input | Type | Base model |
|---|---|---:|
| fixed promotion setup | deterministic | $300 |
| hourly labor rate | deterministic | $18/hour |
| food-cost rate | deterministic | 0.30 |
| customers | uncertain | discrete PMF |
| average spend | uncertain | Normal($22, $2.50²), clipped at $1 |
| redemptions | uncertain, conditional | Binomial(customers, 0.20) |
| labor hours | uncertain, demand-linked | 20 + 0.12 customers + noise |

Monte Carlo does **not** require every input to be random. Fixed assumptions remain important—and must still be validated.

### The deterministic forecast first

At 180 customers, $22 average spend, 36 expected redemptions, and 45 labor hours:

* revenue = $3,960;
* food cost = $1,188;
* labor cost = $810;
* promotion cost = $444;
* profit = **$1,518**.

What does $1,518 hide? Low-demand evenings, unusually high staffing, changing spend, random redemption, and bad combinations of inputs. A point forecast has no probability of loss or downside percentile.

## From one evening to 10,000

Promotion demand has support 120, 150, 180, 210, and 240, with probabilities 0.10, 0.20, 0.35, 0.25, and 0.10. The PMF must sum to one. First draw customers. Then draw

\[
RedeemedOffers\mid Customers=n\sim Binomial(n,0.20).
\]

The number of redemption opportunities depends on arrivals: this is conditional probability made executable. Labor is deliberately linked to demand:

\[
LaborHours=20+0.12Customers+\epsilon,\quad\epsilon\sim Normal(0,2^2).
\]

For a hand-worked evening with 210 customers, $21.40 spend, 39 redemptions, and 47.2 labor hours: revenue is $4,494; food is $1,348.20; labor is $849.60; promotion cost is $456; and profit is **$1,840.20**. This is one plausible evening, not a forecast.

| Scenario | Customers | Spend | Redeemed | Labor hours | Profit |
|---:|---:|---:|---:|---:|---:|
| 1 | 150 | $21 | 26 | 38 | $1,417 |
| 2 | 210 | $23 | 45 | 47 | $1,927 |
| 3 | 120 | $20 | 18 | 34 | $1,116 |

Monte Carlo automates this accounting calculation across many plausible scenarios. The implementation uses `np.random.default_rng(seed)`, never global random state, and vectorized NumPy draws to create an in-memory DataFrame. Same seed, code, and assumptions reproduce results.

## Reading simulated profit

Profit is the output random variable. Its mean describes expected reward; its standard deviation describes spread. Its 5th percentile means 5% of modeled outcomes are at or below that amount; percentiles require no Normal assumption. Decision metrics include:

* probability of loss, \(P(Profit<0)\);
* probability of missing a $500 minimum;
* probability of exceeding $1,000;
* median, 5th percentile, and 95th percentile.

A positive expected profit can coexist with substantial loss probability. That is not automatically a good promotion: operating tolerance, liquidity, constraints, and tail outcomes matter.

The empirical CDF plots \(P(Profit\le x)\). Its height at zero visually estimates loss probability. Minimum and maximum are reported, if at all, cautiously because they depend strongly on simulation count.

## Compare strategies and incremental profit

The no-promotion strategy has lower discrete demand and no discount or setup cost; spending, food cost, and staffing logic remain comparable. For aligned trial rows,

\[
\Delta Profit=Profit_{promotion}-Profit_{baseline},
\]

and \(P(\Delta Profit>0)\) directly answers how often promotion wins. Using the same seed aligns random scenarios where practical (a light use of common random numbers); it does not make the strategies identical.

A fixed-input break-even search finds the first customer count with nonnegative profit, connecting to Chapter 1. Spend and redemption remain uncertain, so a threshold alone cannot describe risk.

## Dependence matters

**Model A—unrealistic independence** gives labor the hours implied by average demand plus noise, regardless of actual customers. **Model B—demand-linked labor** increases hours with customers. Their profit distributions and customer–labor correlations differ.

> Assuming uncertain inputs are independent merely for computational convenience can materially distort risk estimates.

Empirical correlations between customers and labor, customers and revenue, and labor and profit reconnect Chapter 19. They describe simulated co-movement under this model; they do not establish causation.

Conditional masks similarly estimate \(P(Profit<0\mid Customers<150)\) and compare it with unconditional loss risk.

## Sensitivity and scenario analysis

Monte Carlo asks, “What happens under assumed uncertainty?” One-at-a-time sensitivity asks, “Which assumptions matter most?” The experiment runs low/base/high values for redemption probability, average-spend mean, food-cost rate, labor hours per customer, and setup cost, using fixed seeds for reproducibility. A large expected-profit range identifies an assumption worth validating—not a causal effect.

Scenario analysis instead constructs a few deliberate pessimistic, base, and optimistic cases. Both are useful; thousands of probabilistic trials do not make deliberate scenarios obsolete.

Runs of 100, 1,000, and 10,000 trials illustrate declining Monte Carlo noise. Estimates need not approach their large-run values monotonically.

### The simulation is only as good as the model

Validation checks the PMF, parameter ranges, support, nonnegative redemptions, `redeemed <= customers`, labor bounds, finite values, costs, and the accounting identity. Simulation code is still analytical data engineering and must reconcile like the Chapter 12 pipeline.

**Simulation error** comes from a finite number of random trials and can be reduced by more trials. **Model error** comes from a wrong demand distribution, redemption rate, dependence structure, tails, stale data, structural change, omitted input, or bug; more trials cannot repair it.

The model treats parameters such as 0.20 as fixed. It therefore captures **outcome uncertainty**, not necessarily **parameter uncertainty**. A parameter may itself have been estimated imprecisely. Learning unknown parameters from samples belongs to Part IV, not this chapter.

## Recommendation discipline

A recommendation must state: (1) the decision, (2) expected and incremental evidence, (3) loss probability and downside percentile, (4) the most sensitive assumption, (5) what real data must validate, and (6) what cannot be concluded. Do not automatically choose the larger mean.

The simulation cannot prove profitability; its probabilities are conditional on assumptions; simulated demand is not observed future demand; omitted dependence can alter risk; simulation alone establishes no causal promotion effect; and parameter values may be uncertain.

## Common misconceptions

1. **“Monte Carlo predicts the future.”** It generates possibilities under assumptions.
2. **“100,000 trials means accuracy.”** More trials reduce simulation noise, not model error.
3. **“Random inputs should be independent.”** Operational relationships often create dependence.
4. **“Expected profit is enough.”** Downside, spread, constraints, and tails also matter.
5. **“Simulated data is observed data.”** The analyst's probability model generated it.
6. **“Correct implementation proves realistic assumptions.”** Correctness and validity differ.
7. **“Stable output proves parameters are known.”** Parameters can remain uncertain.

## Mastery checkpoints

### Concept

Define Monte Carlo simulation and one trial. Distinguish an outcome from a forecast, simulation error from model error, and expected profit from a complete decision. Interpret a 5th percentile. Explain why dependence matters and why more trials cannot fix assumptions.

### Hand-worked

For 150 customers, $21 spend, 26 redemptions, 38 labor hours, 30% food cost, $18 labor, $300 setup, and $4 per redemption, calculate revenue, food cost, labor cost, promotion cost, and profit by hand.

### Probability and dependence

Using Boolean masks, calculate \(P(Profit<0)\), \(P(Profit>1000)\), and \(P(Profit<0\mid Customers<150)\). Why can independent labor misrepresent risk when busy evenings require more staff?

### Strategy and sensitivity

Given both summaries, recommend a strategy using expected profit, loss probability, downside percentile, and \(P(\Delta Profit>0)\)—never the mean alone. If moving redemption from 0.20 to 0.30 changes profit dramatically while spend SD barely matters, validate redemption first.

### Execution

Define input models; simulate conditional Binomial redemption; encode labor dependence; calculate profit vectorized; summarize percentiles and risks; compare strategies and incremental advantage; run sensitivity; and reproduce results from a fixed seed.

## W&M preparation connection

* **BUAD 512A — Probability & Statistics with R:** integrates distributions, conditional probability, expectation, variance, and dependence.
* **BUAD 512B — Business Modeling with Python:** turns assumptions into reproducible vectorized NumPy and pandas experiments.
* **BUAD 5112 — Competing Through Business Analytics:** compares decisions under uncertainty and communicates downside and credibility conditions.

This independent preparation chapter does not reproduce any W&M course.

## Part III cumulative review

* **Chapter 14:** events define uncertain possibilities.
* **Chapter 15:** conditional probability updates uncertainty when information is known.
* **Chapter 16:** random variables turn uncertain outcomes into numbers.
* **Chapter 17:** distributions allocate probability.
* **Chapter 18:** expectation and variance describe center and spread.
* **Chapter 19:** covariance and correlation describe co-movement and dependence.
* **Chapter 20:** Monte Carlo combines these pieces in an executable decision model.

> So far, the probability model has mostly been treated as known. In real analytics, we usually do not know the true probabilities or parameters. We observe a sample and try to learn about the larger process that generated it.

That question begins Chapter 21.

Run this chapter:

```bash
python3 -m analytics_foundations chapter-20
```

> **Central outcome:** Monte Carlo simulation converts explicit probability assumptions into a distribution of business results. It compares strategies using expected outcomes, downside risk, percentiles, conditional probabilities, and sensitivity—but decision quality still depends on credible assumptions.
