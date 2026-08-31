# Chapter 21 — Samples Tell Stories

> **Central outcome:** A sample is evidence about a larger population or process, not the population itself. Sample statistics naturally vary, and whether they support broader conclusions depends on how observations were selected, measured, and connected to the target population.

## From probability to statistics

In Part III, we often assumed a model such as

$$X\sim N(12,2^2)$$

and asked, “What outcomes does this model produce?” Real analytics often begins instead with observations such as

```text
11.7, 13.2, 10.9, 12.5, 14.1, ...
```

and asks, “What can these observations tell us about the unknown process that generated them?”

```text
Probability:
Model → Data

Statistics:
Data → Model / Population
```

The second arrow is harder: many possible populations could have produced the same sample. This chapter therefore stops short of formal sampling distributions, confidence intervals, and hypothesis tests.

## James River Restaurant Group

James River Restaurant Group wants to understand wait time across all Friday dinner parties this quarter. Measuring every relevant party may be costly or impossible, so management observes a sample from selected evenings.

**Target population:** all seated Friday dinner parties at all five James River locations during Q1 2026.

**Observational unit and grain:** one seated party's wait-time observation. Do not say a sample contains “500” unless you can say **500 what**. A customer, party, transaction, restaurant-day, employee, or product implies a different grain (Chapter 12).

The target definition specifies the unit, quarter, service, locations, and inclusion rule. “What is our average wait?” does not.

The executable experiment creates a clean, deterministic synthetic finite population of 5,000 parties with variation by location, service period, and party size. Its fields are `observation_id`, `date`, `location_id`, `wait_minutes`, `party_size`, `reservation`, `arrival_hour`, and `service_period`. We reveal the complete population only for teaching. In real inference, its parameters usually would not be known.

## Population, sample, and census

A **population** is the full set of observations or outcomes about which we want to make a statement. Population does not mean only people. Examples include all promotion transactions, all products from a line, or all service calls under comparable conditions.

A **finite population** is bounded, such as all 8,214 transactions last quarter. A **conceptual process population** is the outcomes a continuing process could generate, such as all future Friday waits under the current operating process. Inference often concerns that continuing process.

A **sample** is the subset actually measured and analyzed. We write $N$ for finite population size and lowercase $n$ for sample size.

A **census** measures every member of a finite target population. It may still be unavailable or insufficient because of cost, time, inaccessible observations, measurement burden, a changing process, or future outcomes that do not yet exist. Sampling is often unavoidable.

## Parameters and statistics

A **parameter** is a numerical characteristic of a population and is usually unknown. A **statistic** is calculated from observed sample data.

| Population quantity | Sample quantity |
|---|---|
| mean $\mu$ | mean $\bar{x}$ |
| standard deviation $\sigma$ | standard deviation $s$ |
| proportion $p$ | proportion $\hat p$ |

> Statistics are observable; parameters are usually what we want to learn about.

For sample waits $x_1,\ldots,x_n$,

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.$$

The sample mean is an estimate, not automatically the population mean. Sample variability is

$$s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}.$$

Standard sample variance uses $n-1$ because it estimates population variability from a sample; a deeper degrees-of-freedom explanation can wait.

For the business question “What fraction of target parties waited more than 20 minutes?”, define the Chapter 16-style indicator

$$Y=\begin{cases}1,&\text{wait exceeded 20 minutes}\\0,&\text{otherwise.}\end{cases}$$

Then $\hat p=(\text{number with }Y=1)/n$ estimates unknown population proportion $p$.

An **estimator** is a rule, such as $\bar X$ as the procedure that calculates a sample mean. An **estimate** is one realized value, such as $\bar x=16.4$.

## Hand-worked partial views

For the tiny population $\{10,12,14,16,18,20\}$, $\mu=15$. Sample $\{10,14,18\}$ gives $\bar x=14$, while $\{12,16,20\}$ gives $\bar x=16$. Neither equals $\mu$, but both are plausible estimates from partial information. The deliberately low sample $\{10,12,14\}$ gives $\bar x=12$ and illustrates how systematic selection can push an estimate.

## One sample is one story

One random sample might give $\bar x=16.8$, another $15.9$, and a third $17.3$ even though the population did not change.

> Sample statistics vary from sample to sample because the sample changes.

The difference $\bar x-\mu$ is the **sample estimation error for this one sample**. It is not a formal standard error. Chapter 22 will study the pattern of repeated statistics; this chapter draws only a handful.

## How selection creates evidence

In idealized **simple random sampling**, every target-population observation has a known and comparable chance of selection. The practical goal is not to systematically favor certain observations. In Python:

```python
sample = population.sample(n=40, random_state=21)
```

This demonstration is possible because the synthetic teaching population is complete. Most finite-population sampling is **without replacement**, so a member appears at most once. Sampling **with replacement** permits reselection and is useful in simulation, but finite-population correction is beyond this chapter.

A **sampling frame** is the operational list or mechanism from which selection actually occurs. The target may be all Friday dinner parties while the frame is the reservation/seating system. If it omits walk-ins, it undercovers the target even if records are randomly selected.

### Selection problems

**Sampling bias** occurs when selection systematically overrepresents or underrepresents parts of the target population.

- A **convenience sample** surveys whoever is easiest to reach, perhaps early diners. Convenience is not representativeness.
- **Voluntary response** can overrepresent unusually satisfied, dissatisfied, or engaged review writers. Their $\hat p$ need not represent all customers.
- **Undercoverage** occurs when a five-location question uses one location, or Friday dinner is represented by weekday lunch.
- **Nonresponse** matters when appropriately selected nonresponders differ systematically from responders.

> **A large biased sample can still be misleading. Increasing sample size reduces random variation, but does not automatically remove bias.**

Management might survey only online reviewers, pre-5:30 diners, one location, or coupon takers. Each selection mechanism can make the sample systematically different from the target.

### Random variation is not selection bias

A properly selected random sample differs from the population by chance: **random sampling variation**. A biased mechanism pushes results systematically: **selection bias**. Larger random samples generally contain more information and tend to fluctuate less, but size does not repair a systematic miss.

The experiment compares five samples each for $n=10$, $40$, and $200$ without deriving $\sigma/\sqrt n$. It then compares an unbiased $n=40$ sample with 500 convenient Harbor observations. The large restricted sample should be worse because it answers a narrower question very consistently.

### Measurement and dependence

A representative sample can still produce bad evidence when measurement is flawed. Host recollections are not the same as timestamped waits. The former is a **measurement problem**; omission of late diners is a **selection problem**.

Separate rows are not automatically independent. Forty customers from two giant parties may not behave like 40 independent units. The same warning applies to repeated purchases by one customer, transactions from one store-day, or repeated measurements of one employee. This chapter develops awareness, not clustered-sampling formulas.

## Composition and stratification intuition

Population and sample location shares can expose obvious coverage failures. Matching one marginal percentage does not by itself prove representativeness, but a sample that is 100% Harbor obviously cannot mirror a five-location population.

When locations differ, an analyst might deliberately take 10 parties from each. This is **stratified-sampling intuition**: business structure helps ensure coverage. The executable example is unweighted and introduces no formal formulas.

## Data-generating process and generalization

The **data-generating process** is the mechanism producing observations: customer arrivals, demand, staffing, survey response, and transaction logging. Inference needs assumptions about that process.

**Generalization** extends a sample conclusion to a broader target population or process. It is more defensible with:

- a relevant, precise target population;
- an appropriate selection process and frame;
- reliable measurement;
- adequate sample size;
- awareness of clustering or dependence; and
- a stable business process connecting sample and target.

These restaurant waits are **observational data**: the analyst records what occurred. In **experimental data**, an analyst assigns a treatment or condition. A sample-based association is not automatically causal.

Say, “The sample mean is 16.4 minutes.” Do not say, “The restaurant's true mean is 16.4 minutes,” unless the relevant population was fully measured.

## Descriptive and inferential statistics

**Descriptive statistics** summarize observed data: a sample mean, median, histogram, or sample SD. **Inferential statistics** use sample evidence to reason about unknown population quantities. Confidence intervals, hypothesis tests, and regression inference are previews only; later chapters teach them.

Conceptually, an estimation procedure is biased when repeated use systematically misses the parameter in one direction. Formal estimator properties also wait for later chapters.

## Executable workflow

Run:

```bash
python3 -m analytics_foundations chapter-21
```

The pandas/NumPy workflow:

1. constructs and inspects the deterministic population table;
2. defines the target and observational unit;
3. reveals synthetic $\mu$, $\sigma$, and $p$ for teaching;
4. draws a reproducible random sample of $n=40$;
5. calculates $\bar x$, $s$, $\hat p$, and $\bar x-\mu$;
6. compares five samples and three sample sizes;
7. constructs 500 observations restricted to Harbor;
8. compares location composition;
9. demonstrates 10 observations per location; and
10. creates five figures.

### Visual evidence

- `chapter-21-population-vs-sample.png`: imperfect resemblance.
- `chapter-21-several-samples.png`: five means against $\mu$.
- `chapter-21-sample-size.png`: larger-sample stability intuition.
- `chapter-21-size-vs-bias.png`: population, random 40, and biased 500.
- `chapter-21-composition.png`: population, random, and restricted location shares.

None is a formal distribution of a statistic.

## Common misconceptions

1. **“A sample is just a smaller dataset.”** It is a subset intended to inform a defined population or process.
2. **“A large sample must be representative.”** A large biased sample can precisely describe the wrong subset.
3. **“A difference from $\mu$ means something went wrong.”** Random samples naturally vary.
4. **“Random sampling guarantees a perfect match.”** It controls systematic selection but not chance variation.
5. **“Every row is independent.”** Rows can be clustered or repeated measures.
6. **“A statistic is the parameter.”** A statistic estimates an unknown population quantity.
7. **“More data fixes everything.”** It does not fix bad selection, measurement, or target definition.

## Mastery checkpoints

### Concept checkpoint

Explain population, sample, parameter, statistic, and descriptive versus inferential statistics. Why can random samples have different means? Why does size not eliminate bias? What would “representative” mean for this restaurant question?

### Parameter/statistic checkpoint

Classify and give notation where appropriate:

- all target customers' average wait ($\mu$, parameter);
- 50 sampled customers' average wait ($\bar x$, statistic);
- true churn proportion ($p$, parameter);
- observed sample churn proportion ($\hat p$, statistic);
- population standard deviation ($\sigma$, parameter); and
- sample standard deviation ($s$, statistic).

### Sampling-design checkpoint

Identify each problem:

- only online reviewers: voluntary-response/selection bias;
- all-store wait estimated from one location: undercoverage;
- reservation records sampled but walk-ins missing: frame undercoverage;
- 1,000 customers sampled but waits recalled by staff: measurement problem.

### Size-versus-bias checkpoint

Which evidence is more defensible for the target: 50 randomly selected target parties or 5,000 convenient early-evening parties? The random sample may be more defensible because the convenience sample systematically misses relevant service periods. Size alone does not settle the question.

### Execution checkpoint

Define a target and observational unit; draw a reproducible random sample; calculate sample statistics; compare several samples; construct a biased sample; compare composition; and state what can and cannot be generalized.

### Interpretation checkpoint

- If $\bar x=17.2$, is $\mu$ exactly 17.2? **No; 17.2 is an estimate.**
- If another sample gives 16.5, did the process necessarily change? **No; random samples naturally differ.**
- If 10,000 app users exclude phone customers, is the estimate automatically reliable? **No; the target may be undercovered.**

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** population, sample, parameter, and statistic prepare learners for later sampling distributions, estimation, confidence intervals, tests, ANOVA, and regression inference.
- **BUAD 512B — Business Modeling with Python:** pandas and NumPy make sampling, summaries, and reproducible experiments executable.
- **BUAD 5112 — Competing Through Business Analytics:** recommendations depend on whether observations represent the population and decision context—not merely on what columns exist.

This independent chapter prepares foundations; it does not reproduce any W&M course.

## The unresolved question

Suppose we repeatedly took random samples of size $n=40$ and calculated $\bar X$ each time. Those means would vary. But:

> **Is there a predictable pattern to that variation?**

That is the central question of Chapter 22. We do not answer it here.
