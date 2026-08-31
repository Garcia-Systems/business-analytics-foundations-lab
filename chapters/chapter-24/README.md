# Chapter 24 — Hypothesis Testing

> Our service standard says average Friday dinner wait time should be no more than 15 minutes.

**Business question:** Does the sample provide evidence that the true mean wait time is above 15 minutes?

Hypothesis testing asks how compatible the observed sample is with a specific population claim. The test statistic expresses the observed difference relative to sampling uncertainty, while the p-value quantifies how extreme that evidence would be under the null model. A decision still requires effect size, error consequences, sampling quality, and business importance.

## From claim to hypotheses

Translate the question before inspecting the answer:

\[
H_0:\mu=15 \qquad H_A:\mu>15.
\]

The **null** is the benchmark used to generate the reference distribution; it is not automatically what we believe. The **alternative** is the direction investigated. Equality belongs in the null. Hypotheses concern the population parameter \(\mu\), not the observed statistic \(\bar x\): \(H_0:\bar x=15\) is wrong.

Three questions produce different alternatives:

| Form | Alternative | Business question |
|---|---|---|
| right-tailed | \(\mu>\mu_0\) | Is the mean above the threshold? |
| left-tailed | \(\mu<\mu_0\) | Is the process faster than the benchmark? |
| two-sided | \(\mu\ne\mu_0\) | Has the mean changed either way? |

Do not write vague hypotheses such as “wait time is bad.” Choose direction from the business question **before** seeing results; switching afterward is data snooping.

## The null model and signal-to-noise statistic

If \(H_0:\mu=15\) were true, sample means would still vary. Chapter 22 supplied that machinery. We now ask: *under a true population mean of 15, how unusual is the observed mean?* When population SD \(\sigma\) is unknown,

\[
t=\frac{\bar x-\mu_0}{s/\sqrt n}=\frac{\text{signal}}{\text{estimated noise}},\qquad T\sim t_{n-1}.
\]

The numerator is the observed difference; the denominator is its estimated standard error. Thus, \(t\) says how many estimated standard errors the sample mean lies from the null value. We use t again because \(s\) replaces unknown \(\sigma\); Chapter 23 developed this reference distribution.

### Hand-worked example

With \(\bar x=17,\mu_0=15,s=8,n=64\),

\[
SE=8/\sqrt{64}=1,\qquad t=(17-15)/1=2.
\]

The observed mean is **2 estimated standard errors above** the null. With 63 df the right-tail probability is about 0.025. The same mechanics applied to \(\bar x=102,\mu_0=100,s=12,n=36\) give \(SE=2\) and \(t=1\).

## The p-value: a conditional tail probability

The p-value is the probability, **assuming the null hypothesis and test model are correct**, of obtaining a statistic at least as extreme as observed in the direction specified by the alternative:

\[
p=P(T\ge t_{obs}\mid H_0)\quad\text{(right)},\qquad
p=P(T\le t_{obs}\mid H_0)\quad\text{(left)},
\]
\[
p=2P(T\ge |t_{obs}|)\quad\text{(symmetric two-sided)}.
\]

**A p-value is not \(P(H_0\mid data)\).** A value of 0.03 does not mean a 3% chance that the null is true, a 97% chance the alternative is true, or a vague 3% chance the result “happened by chance.” It describes data under a stipulated null model.

The implementation deliberately exposes every calculation before verifying it:

```python
n = len(x)
mean = x.mean()
sd = x.std(ddof=1)
se = sd / np.sqrt(n)
t_stat = (mean - null_mean) / se
p_right = scipy.stats.t.sf(t_stat, df=n - 1)
p_left = scipy.stats.t.cdf(t_stat, df=n - 1)
p_two = 2 * scipy.stats.t.sf(abs(t_stat), df=n - 1)
```

`scipy.stats.ttest_1samp(..., alternative=...)` is a check, not a hiding place for the manual calculation. For a positive statistic, symmetry makes the two-sided p-value twice the upper-tail p-value. A one-sided test is not a trick for a smaller p-value; it asks a different question.

## Alpha, decisions, and truth

Choose \(\alpha\), often 0.05 in examples, before testing. If \(p\le\alpha\), **reject \(H_0\)**; otherwise **fail to reject \(H_0\)**. Never “accept the null”: insufficient evidence against 15 does not prove the mean equals 15. For example, \(p=.03\) at \(\alpha=.05\) supports “the sample provides statistically significant evidence that the population mean exceeds 15,” not “the null is false.” At \(p=.18\), the sample provides insufficient evidence of an exceedance—not proof of equality.

The equivalent critical-value view chooses \(t^*\) with \(P(T\ge t^*)=\alpha\) and rejects when \(t_{obs}\ge t^*\). Two-sided rejection regions split alpha across both tails. The middle is a **non-rejection region**, not proof of acceptance.

| Reality | Fail to reject | Reject |
|---|---|---|
| \(H_0\) true | correct decision | **Type I error** |
| meaningful alternative true | **Type II error** | correct detection (power) |

Type I error means spending on staffing changes after concluding waits exceed 15 when the true mean is 15; its designed probability is \(\alpha\), under assumptions. Type II error means failing to act when, say, the true mean is 18; denote its probability lightly by \(\beta\). **Power \(=1-\beta\)** is rejection probability at a specified alternative. Power generally rises with larger true effects, larger samples, smaller variability, or larger alpha.

Lower alpha reduces Type I risk but, with sample size and effect fixed, makes rejection harder and can raise Type II risk. Which error costs more is a business question, so 0.05 is not universal.

## Evidence strength is not business importance

A result is statistically significant at alpha when its p-value is at or below alpha. That says nothing automatically about importance. With a huge sample, \(\bar x=15.2\) may yield a tiny p-value, but 0.2 minutes is only 12 seconds. Is that operationally meaningful? Maybe not.

Always report the raw effect \(\bar x-\mu_0\) in minutes with the p-value. The statistic is sensitive to effect, variability, and sample size: holding a one-minute effect and \(s=8\) fixed, increasing \(n\) from 16 to 64 to 400 shrinks SE and grows t. Holding effect and n fixed, larger s increases noise and weakens evidence. Therefore lower p-values do not necessarily mean larger business effects. Evidence also changes continuously: 0.049 and 0.051 are not fundamentally different even when an organization needs a threshold.

## Confidence-interval connection

For a matching two-sided test of \(H_0:\mu=\mu_0\) at \(\alpha=.05\), the null lies outside the 95% two-sided t interval exactly when the test rejects. The experiment calculates both and checks this numerical consistency. Do not extend that equivalence carelessly to mismatched one-sided procedures; a one-sided test naturally corresponds to a one-sided bound.

## Simulations and diagnostics

The analytical test remains primary. Fixed-seed simulation makes long-run ideas tangible:

* Samples generated with true \(\mu=15\) reject near 5% of the time at \(\alpha=.05\): Type I error is a repeated-procedure property, not a promise about one run.
* Samples generated with true \(\mu=17\) show higher rejection probability as n grows from 16 to 64 to 256.
* At fixed n, true means 15.5, 16, 17, and 19 illustrate rising simulated power as departure grows.
* Appending one extreme wait changes both mean and SD. Inspect the distribution before trusting a mean-based test; do not blindly use a normality test as a gatekeeper.

Assumptions include a clearly defined target population, reasonably representative and independent observations, reliable measurement, a stable process, and no severe skew/outliers in very small samples. Direction must precede inspection. Large samples reduce sampling uncertainty but do not repair bias, measurement error, misspecification, or an irrelevant population.

Testing 20 unrelated metrics at 5% can produce small p-values by chance. Trying many thresholds, retaining the smallest p-value, choosing direction after inspection, or checking repeatedly until significance undermines the advertised interpretation. Repeated testing requires additional care.

## Mastery checkpoints

1. Write parameter hypotheses: order value changed from $32 (two-sided); defect rate above 2% (right-tailed); handling time reduced below 8 minutes (left-tailed). Where does equality belong?
2. Interpret \(SE=2,t=1\) for the 102-versus-100 example in estimated-standard-error units.
3. A test gives p=.04. Is there a 4% probability the null is true? **No**—state the correct conditional tail interpretation.
4. With p=.12 and alpha=.05, choose: accept, prove no effect, or **fail to reject**.
5. With n=100,000, a 0.1-minute increase, and p<.001, is there automatically an operational problem? **No**; assess the six-second effect in context.
6. Identify the restaurant Type I and Type II errors. If the true mean is 18, what generally happens to power as sample size grows?
7. Execute the workflow: state hypotheses; manually compute SE, t, and the correct tail; verify with SciPy; compare with alpha; report the raw effect and CI; then simulate Type I error and power.

Run it with:

```bash
python3 -m analytics_foundations chapter-24
```

The figures show the null tail, one- versus two-sided extremeness, repeated null rejections, power versus sample size, statistical versus business significance, and the confidence-interval/null-value link.

## Preparation connection and transition

This chapter prepares for core inference in **BUAD 512A**, executable NumPy/SciPy modeling in **BUAD 512B**, and the evidence-versus-consequences judgment central to **BUAD 5112**. It is independent preparation and does not reproduce any W&M course.

So far the claim involved one mean, \(\mu=15\). New questions ask whether Location A waits longer than Location B, whether a promotion changed spending, or whether three locations have equal service times. How can we tell whether group differences exceed ordinary sampling variation? That is Chapter 25. No two-sample test or ANOVA is performed here.
