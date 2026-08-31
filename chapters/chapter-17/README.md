# Chapter 17 — Distributions

> **Central question:** How can a few parameters describe an entire family of uncertain outcomes?

The path is **business process → random variable → distribution choice → parameters → shape → probability calculation → simulation → model check → interpretation**.

> A probability distribution is a model for how uncertainty is allocated across the possible values of a random variable. Its parameters control the model's shape and business meaning.

Specifying hundreds of probabilities by hand is impractical. Named mathematical families provide reusable models. The notation (X\sim D) simply means “random variable (X) follows distribution (D).” A **parameter** is a numerical value determining a property of the probability model; an **observation** is one realized value. Parameters encode business claims, not merely mathematical decoration.

## Bernoulli: one binary trial

For one customer's offer response,

\[
X=\begin{cases}1,&\text{redeems}\\0,&\text{does not redeem}\end{cases},\qquad X\sim\operatorname{Bernoulli}(0.20).
\]

Thus (P(X=1)=.20) and (P(X=0)=.80). This is Chapter 16's indicator variable as a named model. “Success” only labels 1; it need not be desirable (fraud can be success). Purchase, defect, churn, late delivery, click, and fraud indicators can all be Bernoulli when one binary trial is appropriate.

After seeing the cases, the compact PMF is

\[
P(X=x)=p^x(1-p)^{1-x},\quad x\in\{0,1\}.
\]

For (x=1), (p^1(1-p)^0=p); for (x=0), (p^0(1-p)^1=1-p). `rng.binomial(1, .20, size)` simulates zeros and ones; the sample proportion of ones can be compared with (p), but need not equal it.

## Binomial: successes across repeated trials

If 20 independently modeled customers each redeem with probability .20 and (X) counts redemptions,

\[
X\sim\operatorname{Binomial}(n=20,p=.20).
\]

Bernoulli models one binary trial; Binomial counts successes across (n) such trials. Its assumptions are **(1)** fixed (n), **(2)** two modeled outcomes per trial, **(3)** common (p), and **(4)** independence under the model.

\[
P(X=k)=\binom nk p^k(1-p)^{n-k},\qquad \binom nk=\frac{n!}{k!(n-k)!}.
\]

Here (p^k) contributes the successes, ((1-p)^{n-k}) the failures, and (\binom nk) counts their arrangements. With three customers and two redemptions, `RRN`, `RNR`, and `NRR` give (\binom32=3).

### Worked calculations

For (X\sim\operatorname{Binomial}(5,.20)),

\[
P(X=2)=\binom52(.20)^2(.80)^3=10(.04)(.512)=0.2048.
\]

SciPy verifies this with `binom.pmf(2, 5, .20)`. Exact, cumulative, and tail questions differ:

```python
binom.pmf(4, n, p)       # P(X=4)
binom.cdf(4, n, p)       # P(X<=4)
binom.sf(3, n, p)        # P(X>3) = P(X>=4)
```

The survival function is (P(X>x)). For the hand example,

\[
P(X\ge2)=1-P(X\le1)=1-\texttt{binom.cdf(1,5,.20)}=0.26272.
\]

Holding (n) fixed while changing (p) shifts mass toward larger success counts as (p) rises. Increasing (n) expands the support (0,\ldots,n) and changes the shape. We defer formulas for center and variability to Chapter 18.

## Uniform: equal density on an interval

Model a shuttle wait as (Y\sim\operatorname{Uniform}(0,10)). Equal-length intervals have equal probability, and

\[
f_Y(y)=\begin{cases}\frac1{b-a},&a\le y\le b\\0,&\text{otherwise.}\end{cases}
\]

Here density is .1 probability per minute. **Density is not probability:** (f_Y(5)=.1), but (P(Y=5)=0). Probability is area (a Chapter 5 integration connection):

\[
P(2\le Y\le5)=\int_2^5\frac1{10}\,dy=(5-2)(.1)=.30.
\]

SciPy parameterizes endpoints carefully: `uniform(loc=a, scale=b-a)`, not `scale=b`. Thus `dist = uniform(loc=0, scale=10)` and `dist.cdf(5)-dist.cdf(2)` also give .30.

## Normal: bell-shaped continuous variation

Suppose order-preparation time is approximated by

\[
T\sim\mathcal N(\mu=12,\sigma^2=2^2).
\]

This is an assumption, not a statement that all business data is normal. The curve is symmetric and bell shaped. (mu) shifts its location; larger (sigma) spreads density more widely. Common notation puts variance (sigma^2) second, while SciPy's `norm(loc=mu, scale=sigma)` expects the standard deviation.

For recognition—not memorization—the PDF is

\[
f(x)=\frac1{\sigma\sqrt{2\pi}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
\]

(x-\mu) measures displacement, (sigma) scales it, squaring gives symmetry, and the Chapter 2 exponential makes density decline away from the center. The model extends in both directions, assigning tiny probability to impossible negative preparation times. A useful approximation need not be a literal description.

```python
from scipy.stats import norm
dist = norm(loc=12, scale=2)
dist.cdf(15)                       # P(T<=15), about .9332
dist.cdf(14) - dist.cdf(10)        # P(10<=T<=14), about .6827
dist.sf(15)                        # P(T>15), about .0668
```

Tail areas later matter in hypothesis testing; testing is not introduced here. Standardization previews later work:

\[
Z=\frac{X-\mu}{\sigma},\qquad \frac{15-12}{2}=1.5.
\]

Thus 15 minutes is 1.5 standard deviations above the model center, and a normal (X) becomes (Z\sim N(0,1)).

## PMF, PDF, and the common CDF

| Tool | Variable | Value means | Probability calculation |
|---|---|---|---|
| PMF (p_X(x)) | discrete | (P(X=x)), possibly positive | add masses |
| PDF (f_X(x)) | continuous | density, not point probability | (int_a^b f_X(x)dx) |
| CDF (F_X(x)) | either | (P(X\le x)) | accumulate through (x) |

A discrete CDF jumps; a continuous CDF changes smoothly when its density is smooth. Interval probabilities are CDF differences. This is the shared language joining Chapters 16 and 17.

## Choosing and checking a model

* **Bernoulli:** one binary outcome.
* **Binomial:** fixed count of Bernoulli-like trials with stable (p) and plausible independence.
* **Uniform:** continuous interval where equal density is defensible.
* **Normal:** approximately symmetric, bell-shaped continuous variation around a center.

Classification is not mechanical. Parameters are business assumptions: (p=.20) asserts a 20% redemption chance; (n=20) asserts exactly 20 trials; (a=0,b=10) assert the interval; (mu=12,sigma=2) assert location and scale.

A common Binomial (p) is poor when customer segments redeem differently or customers influence one another. Uniform is poor when waits cluster near five minutes. Normal is often poor for strongly right-skewed, nonnegative transaction amounts. A histogram of deterministic preparation observations may be broadly consistent with a proposed density or visibly depart from it, but visual similarity does not prove the model. **A convenient distribution can still be a bad model.** Simulation draws implications of a chosen model; it does not establish realism.

Run the reproducible experiment with:

```bash
python3 -m analytics_foundations chapter-17
```

It uses one fixed NumPy `Generator` to simulate Bernoulli, Binomial, Uniform, and Normal realizations, compares Binomial empirical frequencies to its theoretical PMF, and creates six focused figures.

## Common misconceptions

1. **“A named distribution is a fact about data.”** It is a model or process assumption.
2. **“A PDF value is a probability.”** Continuous probability is interval area.
3. **“Binomial applies to any count.”** Its four trial assumptions must hold.
4. **“Normal means ordinary.”** It names one mathematical distribution.
5. **“A normal-looking sample proves normality.”** Visual resemblance is not proof.
6. **“Simulation proves realism.”** It only shows behavior implied by assumptions.

## Mastery checkpoints

### Concept
Define a parameter and (X\sim D). When is Bernoulli appropriate? How does Binomial relate to it, and what assumptions does it make? Distinguish PMF, PDF, and CDF. Why is density not point probability? What do (mu,sigma) control? Why justify model choice?

### Distribution choice
Choose and explain: one customer purchase (**Bernoulli**); purchasers among 50 independent common-(p) customers (**Binomial**); equal-likelihood arrival in a ten-minute window (**Uniform**); symmetric measurement variation (**Normal may be plausible**); long-right-tail daily revenue (**reject automatic Normal choice**).

### Calculation
Calculate one Bernoulli probability, Binomial PMF, Binomial CDF/tail, Uniform interval area, Normal interval probability, and z-score. Show whether each operation adds mass or finds area.

### Assumptions
Assess a proposed Binomial model when customers influence each other, probabilities vary by customer, or sample size is not fixed. Identify independence, common-(p), and fixed-(n) violations.

### Execution
Simulate all four distributions; compare theoretical and empirical behavior; change one parameter and explain the shape change; calculate a CDF and continuous interval; reject one mismatched model.

### Interpretation
Does tiny Normal probability on negative preparation times automatically make the model useless? **No:** ask whether unrealistic regions matter for the decision. What violates a common 20% redemption probability? **Segments, channel, engagement, location, or timing can produce different probabilities.**

## W&M preparation connection

* **BUAD 512A — Probability & Statistics with R:** named models support later sampling distributions, estimation, testing, ANOVA, and regression inference.
* **BUAD 512B — Business Modeling with Python:** NumPy and SciPy calculate, simulate, visualize, and compare models.
* **BUAD 5112 — Competing Through Business Analytics:** distribution choice expresses process assumptions that affect decision quality.

These are preparation connections only; this independent chapter does not reproduce a William & Mary course. Expected value and formal variability begin in Chapter 18, not here.
