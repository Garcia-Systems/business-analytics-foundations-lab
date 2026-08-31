# Chapter 23 — Estimation & Confidence

Management wants the mean Friday dinner wait time, \(\mu\), for the James River Restaurant Group target population. One random sample gives a **point estimate**, \(\bar x\), but one number cannot say whether 16.4 is meaningfully different from 16.0 or 17.0.

> A confidence interval combines a point estimate with sampling uncertainty. The interval changes from sample to sample while the population parameter remains fixed; confidence describes how often the procedure captures that parameter across repeated samples under its assumptions.

## From an estimate to an interval

The **estimator** \(\bar X\) is the rule before data arrive; the **estimate** \(\bar x\) is its realized value. With unknown population SD, estimate the sampling variation using the sample SD (with `ddof=1`):

\[
\widehat{SE}(\bar X)=\frac{s}{\sqrt n}.
\]

SE estimates how much sample means vary across repeated samples. The memorable interval anatomy is

\[
\text{estimate}\pm\text{critical value}\times SE
=\text{estimate}\pm\text{margin of error}.
\]

Every quantity retains the business unit: the estimate, SE, margin, and endpoints are minutes. Margin of error is the distance from the estimate to either endpoint.

### Normal intuition, step by step

Chapter 17's standard normal and Chapter 22's sampling distribution give \(P(-1.96\le Z\le1.96)\approx.95\). If \(\bar x=16.4,s=6.4,n=64\), then

\[
SE=\frac{6.4}{\sqrt{64}}=0.8,
\qquad ME=1.96(0.8)=1.568,
\]
\[
16.4\pm1.568=[14.832,17.968]\approx[14.83,17.97]\text{ minutes}.
\]

Using this procedure, we obtain a 95% interval from about 14.8 to 18.0 minutes for the population mean. It is a range of means reasonably compatible with the sample and assumptions. Do **not** say there is a 95% probability that fixed \(\mu\) is in this calculated interval.

Hand-worked continuation: \(\bar x=18.2,s=10,n=25\) gives \(SE=10/5=2\), approximate margin \(1.96(2)=3.92\), and \(18.2\pm3.92=[14.28,22.12]\).

## Why Student's t is the default

If \(\sigma\) were genuinely known, use \(\bar x\pm z^*\sigma/\sqrt n\). Usually it is unknown and replacing it with \(s\) adds uncertainty. Student's \(t\) has heavier tails, especially at small degrees of freedom:

\[
T\sim t_\nu,\quad \nu=n-1,
\qquad
\bar x\pm t^*_{n-1}\frac{s}{\sqrt n}.
\]

The implementation obtains `t.ppf(1 - alpha / 2, df=n - 1)`. At \(n=10\), df 9 gives a 95% critical value about 2.262, so a fixed SE gives a wider interval than 1.96. SciPy also gives approximately 2.776 at df 4 and 2.045 at df 29; as df grows, t approaches 1.96. The density figure shows this without deriving the distribution.

## Confidence means repeated-sampling coverage

Imagine repeatedly (1) drawing a random sample of the same size, (2) constructing a 95% interval by the same rule, and (3) recording whether it contains fixed \(\mu\). Samples and intervals differ. Under the method's assumptions, approximately 95% of these intervals contain \(\mu\):

\[
\text{coverage}=\frac{\text{intervals containing }\mu}{\text{intervals constructed}}.
\]

The executable experiment constructs 5,000 intervals from the known teaching population and displays 50; misses use a different marker/line style. **The intervals move. The population parameter does not.** Empirical results need only be near, not exactly, nominal coverage.

## Confidence, precision, and sample size

For one sample, higher coverage costs precision. With fixed \(SE=1\), normal margins are about 1.645, 1.960, and 2.576 for 90%, 95%, and 99%. Thus greater confidence means wider intervals; 99% is not unconditionally “better.” Precision means tightness, not correctness.

Because \(SE=\sigma/\sqrt n\), width shrinks roughly as \(1/\sqrt n\). Quadrupling \(n\) roughly halves the margin; merely doubling it does not. For desired margin \(ME\), a normal planning approximation is

\[
n=\left\lceil\left(\frac{z^*\sigma}{ME}\right)^2\right\rceil.
\]

For 95% confidence, prior \(\sigma\approx8\), and margin at most 1 minute:

\[
n=\left\lceil(1.96\cdot8)^2\right\rceil
=\lceil245.8624\rceil=246.
\]

Prior records, a pilot, or domain knowledge must supply \(\sigma\). This is planning—not a guarantee that the sampling design is valid.

**Confidence intervals do not fix bad data.** A narrow interval around a selection-biased estimate remains misleading. Intervals quantify sampling uncertainty under sampling/model assumptions; they do not automatically cover selection bias, measurement error, leakage, or the wrong target population. Larger \(n\) can make the wrong answer more precise.

## Individuals, proportions, and business meaning

SD describes customers; SE describes a statistic across samples. If \(s=8\) minutes and \(SE(\bar x)=1\) minute, individuals vary on an 8-minute scale while mean uncertainty is on a 1-minute scale. A CI for \(\mu\) asks where the population mean might be, not where the next customer's wait will fall and not where 95% of customers fall. That latter question requires a different procedure such as a prediction interval.

For the proportion waiting over 20 minutes, the intuitive large-sample SE is

\[
SE(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}n},\qquad \hat p\pm z^*SE.
\]

The simple Wald interval can behave poorly at small \(n\) or near 0/1. The experiment therefore reports the better-behaved **Wilson interval** from statsmodels while retaining Wald's formula for intuition.

For a service goal \(\mu\le15\), an interval \([14.8,18.0]\) includes 15 and compatible values above it, so the evidence does not clearly establish that the mean meets the target. An interval \([12.8,14.6]\) entirely below 15 provides stronger evidence that the mean is below it—but this chapter performs no formal test. And although \([16.35,16.45]\) is highly precise, management must ask whether a tenth of a minute matters operationally.

## Assumptions

- The sample appropriately represents the target population.
- Observations are reasonably independent and measurements reliable.
- Severe skew/outliers can undermine small-sample t intervals; larger samples let the CLT help.
- “\(n\ge30\) means everything is fine” is not a valid universal rule.
- Precision is not protection from systematic bias.

Some intervals use resampling rather than formulas, but bootstrap methods are only previewed here. Differences between groups belong in Chapter 25.

## Common misconceptions

1. **“There is a 95% probability fixed \(\mu\) is in this interval.”** The 95% is the procedure's long-run coverage under assumptions.
2. **“99% is always better.”** It covers more often but is wider and less precise.
3. **“A narrow interval proves no bias.”** Biased sampling can be narrowly wrong.
4. **“The interval contains 95% of customers.”** It estimates a parameter, not individual spread.
5. **“SD and SE are interchangeable.”** SD is individual variation; SE is a statistic's sampling variation.
6. **“Large samples fix bad sampling.”** They narrow intervals around biased estimates.
7. **“Outside means impossible.”** An interval is an assumption-dependent inferential procedure, not an absolute possible/impossible list.

## Mastery checkpoints

- **Concept:** Define point estimate, SE, margin, 95% confidence, and precision. Explain why higher confidence widens an interval, larger \(n\) narrows it, individual observations are different, and narrowness does not prove unbiasedness.
- **Arithmetic:** \(\bar x=25,s=12,n=36\) gives \(SE=2\), \(ME=3.92\), and approximate 95% interval \([21.08,28.92]\).
- **Interpretation:** Does CI \([14.2,17.8]\) mean 95% of customers wait there? **No; it is for the population mean.**
- **Confidence:** Across 1,000 repeated 95% intervals, about 95% should cover under assumptions.
- **Precision:** \([17,19]\) is more precise than \([14,22]\), but width alone says nothing about bias.
- **Sample size:** Cutting margin from 4 to 2 requires roughly four times the sample size.
- **Business:** CI \([14.7,16.8]\) cannot establish that \(\mu\le15\), because compatible values exceed 15.
- **Execution:** Calculate an estimate, SE, t critical, and interval; compare confidence and sample sizes; simulate coverage; plan sample size; and interpret in business language.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** intervals connect sampling distributions and SE to later tests, ANOVA, and regression inference.
- **BUAD 512B — Business Modeling with Python:** SciPy, NumPy, pandas, and statsmodels make intervals, coverage checks, and planning reproducible.
- **BUAD 5112 — Competing Through Business Analytics:** managers should receive estimates with uncertainty, not isolated values.

This is independent preparation and does not reproduce a W&M course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-23
```

## The unresolved question

Suppose the operational claim is \(\mu=15\). The interval gives plausible parameter values, but management asks: **How much evidence does this sample provide against the claim that the population mean is 15 minutes?** That is Chapter 24. This chapter calculates no p-value and performs no hypothesis test.
