# Chapter 22 — Sampling Distributions

Chapter 21 drew one sample. Now keep the James River Restaurant Group population fixed, draw random samples of size \(n=40\), and calculate each mean. Results such as 16.2, 16.8, 15.9, and 16.5 differ because each sample contains different parties. Is that random chaos, or does the statistic have a predictable distribution?

> A sampling distribution describes how a statistic would vary across repeated samples from the same population. Its spread is the statistic's **standard error**, and larger samples generally produce more stable estimates.

The full synthetic population of about 5,000 Friday dinner parties is known only for this teaching experiment. **In real inference, population parameters are usually unknown. Here we know them so that we can see whether the sampling theory works.**

## From observations to statistics

- \(X\) is a random customer's wait. It varies because customers differ.
- \(\bar X\) is the mean of a random sample. It varies because samples contain different observations.
- Before collection, \(\bar X\) is a random variable. After collection, \(\bar x\) is its realized numeric value.

The **sampling distribution of a statistic** is its probability distribution across repeated samples drawn with the same process and sample size. It is not the raw population, the values within one sample, or a histogram of a column in one dataset. It is the hypothetical distribution of a *statistic*. Usually we see one sample; simulation makes this repeated-sampling thought experiment visible.

## Three distributions

1. **Population distribution:** all individual waits, \(X\).
2. **Sample distribution:** individual waits observed within one sample.
3. **Sampling distribution:** one \(\bar X\) from each of many samples.

The first chapter figure puts these side by side. The third panel contains means, not customers. Sampling with replacement conveniently represents repeated independent draws from a population model; no finite-population correction is introduced here.

## Build the distribution empirically

The transparent algorithm is:

```python
sample_means = []
for _ in range(n_repetitions):
    sample = rng.choice(population, size=40, replace=True)
    sample_means.append(sample.mean())
```

The implementation vectorizes this as an array with shape `(n_repetitions, sample_size)` and uses `mean(axis=1)` to return one mean per row/sample. The simulated means' average lies near \(\mu\).

## Center: why the sample mean targets the population mean

Using Chapter 18's linearity of expectation,

\[
\bar X=\frac1n\sum_{i=1}^nX_i,
\qquad
E[\bar X]=\frac1n\sum_{i=1}^nE[X_i]
=\frac1n(n\mu)=\mu.
\]

Thus ordinary random sampling centers the sample mean on the population mean. This is an average-over-repetitions statement, not a promise that any one mean equals \(\mu\).

## Spread: standard error

The **standard error of a statistic is the standard deviation of its sampling distribution**. It is not the SD of individual observations. Under independent sampling, Chapter 19's variance rules give

\[
\operatorname{Var}(\bar X)
=\operatorname{Var}\left(\frac1n\sum X_i\right)
=\frac1{n^2}\operatorname{Var}\left(\sum X_i\right)
=\frac1{n^2}(n\sigma^2)=\frac{\sigma^2}{n},
\]

because \(\operatorname{Var}(aX)=a^2\operatorname{Var}(X)\) and independent variances add. Therefore

\[
SE(\bar X)=SD(\bar X)=\frac{\sigma}{\sqrt n}.
\]

Here \(\sigma\) is the population SD of individual waits, \(n\) is sample size, and \(SE(\bar X)\) is the SD of sample means. Simulation uses `sample_means.std(ddof=0)` and compares it with theory.

Precision improves with the square root of sample size—not linearly. Replacing \(n\) by \(4n\) halves SE; replacing it by \(100n\) divides SE by 10. Doubling \(n\) multiplies SE by \(1/\sqrt2\), not one-half. Individuals can vary greatly while their average is stable: if \(\sigma=8\) and \(n=64\), the raw SD is 8 minutes but the mean's SE is 1 minute.

### Hand-worked examples

- \(\sigma=10,n=25\): \(SE=10/5=2\). At \(n=100\), \(SE=10/10=1\).
- \(\sigma=12,n=36\): \(SE=2\). At \(n=144\), \(SE=1\). Halving SE requires four times as many independent observations.
- If \(\sigma\) is unknown, use the practical estimate \(\widehat{SE}(\bar X)=s/\sqrt n\). Chapter 23 will use estimated standard errors to build intervals; this chapter does not build one.

The experiment compares \(n=5,20,50,200\). Their empirical sampling SDs closely track \(\sigma/\sqrt n\), retain approximately the same center, and narrow as \(n\) grows.

## A second statistic: a proportion

For whether a Friday dinner party waits over 20 minutes, let \(X_i\in\{0,1\}\), \(P(X_i=1)=p\), and

\[
\hat p=\frac1n\sum X_i,\qquad E[\hat p]=p,
\qquad SE(\hat p)=\sqrt{\frac{p(1-p)}n}.
\]

At \(p=.20,n=100\), \(SE(\hat p)=\sqrt{.2(.8)/100}=.04\). Simulations at \(n=25,100,400\) show the same shrinking variation.

## Central Limit Theorem

> Under broad conditions, the sampling distribution of the sample mean becomes approximately normal as sample size grows, even when the population itself is not normal.

The experiment deliberately starts with strongly right-skewed lognormal waits. At \(n=1\), the distribution of means resembles that population; at \(n=5\) it is less skewed; at \(n=30\) and \(n=100\) it becomes progressively more bell-shaped and concentrated.

**The CLT does not say raw data becomes normal.** It concerns sampling distributions of statistics such as the mean under appropriate conditions. Nor does \(n\ge30\) guarantee normality: required size depends on skew, heavy tails, extremes, dependence, and the statistic. Thirty is not a law of nature.

Practical conditions include an appropriate sampling process, reasonably independent observations, a stable population/process, and finite variance for the classical theorem. Severe dependence or extreme heavy tails cause trouble. If 100 rows come from only five highly similar restaurant evenings, treating all rows as independent can understate uncertainty: effective information depends on sampling structure, not row count.

## Bias, accuracy, and precision

- **Standard error** describes random sampling variation around an estimator's center.
- **Bias** is systematic displacement of that center from the target.
- **Accurate and precise:** narrow and centered at truth.
- **Accurate on average but imprecise:** wide and centered at truth.
- **Precise but biased:** narrow and centered away from truth.

A large biased sample may have tiny SE and remain wrong. Large \(n\) does not repair selection or measurement bias: **precision is not accuracy**.

## Common misconceptions

1. A sampling distribution is not one sample's values; it is a statistic across samples.
2. The population need not be normal for the CLT to help under suitable conditions.
3. The CLT concerns statistics' sampling distributions, not normalization of raw data.
4. Doubling \(n\) changes SE by \(1/\sqrt2\), not one-half.
5. Large \(n\) reduces random variation, not systematic bias.
6. An observed statistic has one value, but before sampling it is random.
7. Standard error is the SD of a statistic across samples, not raw-observation SD.

## Mastery checkpoints

### Concepts and distribution identification

Explain what a sampling distribution is; whether \(X\), \(\bar X\), or both are random; what SE measures; why SE differs from SD; why size helps; what the CLT says; and why size cannot remove bias. Given plots of the population waits, one sample's waits, and many sample means, identify each and justify the answer by naming its observational unit.

### Formula and simulation

With \(\sigma=18,n=81\), calculate \(SE=2\). To reach SE 1, solve \(18/\sqrt n=1\), giving \(n=324\). Then simulate 5,000 means, report their empirical mean and SD, compare with theory, and repeat at a larger \(n\).

### CLT, bias, and business meaning

Does the CLT say a heavily skewed raw population becomes normal? **No:** the mean's sampling distribution becomes approximately normal under appropriate conditions. A biased estimator centered five units away with SE .3 is more precise than an unbiased estimator with SE 2, but the unbiased estimator is more accurate for the target. Finally, raw wait SD 12 does not make the mean of 100 independent customers' SE 12: it is \(12/\sqrt{100}=1.2\) minutes.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** sampling distributions, SE, and the CLT bridge probability to later inference, ANOVA, and regression.
- **BUAD 512B — Business Modeling with Python:** NumPy makes the repeated-sampling thought experiment executable and checks theory.
- **BUAD 5112 — Competing Through Business Analytics:** managers see one metric, but good judgment asks how it would change under another sample.

This independent preparation chapter does not reproduce any W&M course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-22
```

## The unresolved question

Suppose \(\bar x=16.4\) and its estimated \(SE=0.8\). Another sample would probably give a different mean. **How can one sample estimate and its sampling uncertainty become a range of plausible values for the unknown population mean?** That is Chapter 23. No confidence interval or hypothesis test begins here.
