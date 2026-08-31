# Chapter 18 — Expected Value & Variability

> **Central outcome:** Expected value tells us where an uncertain quantity is centered, while variance and standard deviation tell us how widely outcomes spread around that center. Together they provide a first mathematical language for balancing expected business outcomes against uncertainty and risk.

Chapter 17 gave names to probability distributions. We now ask: **Where is a random variable centered? How much does it vary around that center?** Our path is

> random variable → weighted average → expected value → deviation → squared deviation → variance → standard deviation → business payoff → risk → simulation → interpretation

## From an ordinary average to a weighted average

For observations, the sample mean is

$$\bar{x}=\frac1n\sum_{i=1}^n x_i.$$

In an ordinary average, each observation contributes one equal share. A probability model instead has possible values $x_1,\ldots,x_k$ with possibly unequal probabilities $p_1,\ldots,p_k$. Its probability-weighted average is

$$E[X]=\mu_X=\sum_i x_i p_i=\sum_x xP(X=x).$$

Expected value is the center of the probability distribution in the sense of a probability-weighted average—not necessarily its most likely value.

A fair die makes the distinction vivid:

$$E[X]=\frac{1+2+3+4+5+6}{6}=3.5,$$

although 3.5 cannot be rolled. **Expected value is a model center, not necessarily a possible realization or a prediction of the next trial.** Under suitable assumptions, repeated realizations have an average that tends toward this center. It need not approach monotonically. As a restrained visual metaphor, imagine probability masses on a number line: expectation is their balance point.

## Restaurant special-event package

Management models uncertain package profit:

| Demand state | Profit $x$ | $P(X=x)$ |
|---|---:|---:|
| Weak | -$1,000 | 0.15 |
| Moderate | $500 | 0.35 |
| Strong | $2,000 | 0.30 |
| Very strong | $4,000 | 0.20 |

Every contribution is visible:

$$(-1000)(0.15)=-150,$$
$$500(0.35)=175,$$
$$2000(0.30)=600,$$
$$4000(0.20)=800,$$

so

$$E[X]=-150+175+600+800=\$1{,}425.$$

The units remain dollars: $\text{dollars}\times\text{probability}\rightarrow\text{dollars}$.

### Another PMF

The Chapter 16 customer-count model gives

$$E[X]=80(.10)+120(.20)+160(.35)+200(.25)+240(.10)=162\text{ customers}.$$

This is a probability-model parameter, not an observed day's customer count.

## Functions and linearity of expectation

For a discrete variable and a function $g$,

$$E[g(X)]=\sum_x g(x)P(X=x).$$

If revenue is $R=18X$, then $E[R]=\sum_x18xP(X=x)=18E[X]$. More generally,

$$E[aX+b]=aE[X]+b,\qquad E[X+Y]=E[X]+E[Y].$$

The second rule does **not** require independence; we need not introduce a joint distribution to use it. If fixed daily cost is $1,200$, profit is $P=18X-1200$, hence

$$E[P]=18E[X]-1200=18(162)-1200=\$1{,}716.$$

Directly transforming all five possible customer counts and weighting their profits gives the same \$1,716—an executable consistency check.

## Why center is not enough

Option A pays $1,000 with certainty. Option B pays $-1,000$ or $3,000$, each with probability 0.5. Both have $E[X]=\$1,000$, but their dispersion differs. Likewise:

| Alternative | Outcomes | Expected profit | Standard deviation |
|---|---|---:|---:|
| Stable | $900, $1,100 equally likely | $1,000 | $100 |
| Risky | -$1,000, $3,000 equally likely | $1,000 | $2,000 |

Equal expected reward does not mean equal business risk.

## Deviations, variance, and standard deviation

For mean $\mu=E[X]$, an outcome's signed deviation is $x-\mu$: positive is above the center; negative is below. Signed deviations cannot measure spread because

$$E[X-\mu]=E[X]-\mu=0.$$

Positive and negative deviations cancel. Squaring, $(x-\mu)^2$, prevents cancellation and penalizes larger distances more strongly. Thus

$$\operatorname{Var}(X)=E[(X-\mu)^2]=\sum_x(x-\mu)^2P(X=x).$$

Variance is the probability-weighted average squared distance from expectation. It has squared units—dollars² for profit—which motivates

$$\sigma_X=SD(X)=\sqrt{\operatorname{Var}(X)}.$$

Standard deviation returns to the original units. It is a scale for typical distance from the center, but its precise probabilistic interpretation depends on distribution shape; without a Normal assumption, do not claim that a fixed proportion must lie within one SD.

### Essential hand calculation

For $P(X=0)=.25$, $P(X=10)=.50$, and $P(X=20)=.25$:

| $x$ | $P(X=x)$ | $xP(X=x)$ | $x-\mu$ | $(x-\mu)^2$ | $(x-\mu)^2P(X=x)$ |
|---:|---:|---:|---:|---:|---:|
| 0 | .25 | 0 | -10 | 100 | 25 |
| 10 | .50 | 5 | 0 | 0 | 0 |
| 20 | .25 | 5 | 10 | 100 | 25 |
| **sum** | **1** | **10** | | | **50** |

Therefore $E[X]=0(.25)+10(.50)+20(.25)=10$, the deviations are $-10,0,10$,

$$Var(X)=(-10)^2(.25)+0^2(.50)+10^2(.25)=50,$$

and $SD(X)=\sqrt{50}\approx7.07$.

## Computational shortcut

Expanding the conceptual definition gives

$$E[(X-\mu)^2]=E[X^2-2\mu X+\mu^2]
=E[X^2]-2\mu E[X]+\mu^2=E[X^2]-\mu^2.$$

Thus

$$Var(X)=E[X^2]-(E[X])^2,\qquad E[X^2]=\sum_xx^2P(X=x).$$

For the three-value example, $E[X^2]=0+50+100=150$, so $150-10^2=50$. **$E[X^2]\ne(E[X])^2$ in general; their difference is variance.** The shortcut is convenient, while squared deviations explain the meaning.

## Transformations

$$E[aX+b]=aE[X]+b,$$
$$Var(aX+b)=a^2Var(X),\qquad SD(aX+b)=|a|SD(X).$$

Adding a constant shifts every outcome and its mean equally, leaving deviations unchanged. Hence $Y=X+100$ has $Var(Y)=Var(X)$. Scaling $Y=2X$ doubles deviations, making $Var(Y)=4Var(X)$ and $SD(Y)=2SD(X)$. This also explains squared variance units. For $Y=3X+5$ in the hand example, $E[Y]=35$ and $Var(Y)=9(50)=450$.

## Connections to named distributions

For $X\sim Bernoulli(p)$,

$$E[X]=0(1-p)+1p=p,$$
$$Var(X)=(0-p)^2(1-p)+(1-p)^2p=p(1-p).$$

For $X\sim Binomial(n,p)$, viewed as a sum of $n$ independent Bernoulli indicators,

$$E[X]=np,\quad Var(X)=np(1-p),\quad SD(X)=\sqrt{np(1-p)}.$$

The expectation-of-a-sum rule itself needs no independence. Variance of sums becomes more interesting for dependent variables; Chapter 19 will explain why, without our beginning that topic here.

For $X\sim N(\mu,\sigma^2)$, $E[X]=\mu$, $Var(X)=\sigma^2$, and $SD(X)=\sigma$. For $X\sim Uniform(a,b)$,

$$E[X]=\frac{a+b}{2},\qquad Var(X)=\frac{(b-a)^2}{12}.$$

For continuous variables, probability-weighted sums become the probability-weighted integrals developed in Chapter 5:

$$E[X]=\int_{-\infty}^{\infty}xf_X(x)\,dx,$$
$$Var(X)=\int_{-\infty}^{\infty}(x-\mu)^2f_X(x)\,dx.$$

These formulas are a conceptual preview, not an integration detour.

## Risk, reward, and organizational preference

Is a plan with expected profit $1,100 and SD $1,600 automatically better than one with expected profit $1,000 and SD $300? No. Downside exposure, cash available, repeatability, risk tolerance, consequences of loss, and model reliability matter. **Expected value summarizes reward; standard deviation summarizes one dimension of risk. Neither alone makes the decision.** Monetary expected value is not the same as organizational preference; a catastrophic loss can threaten survival.

A 2% chance of a $50,000 loss and otherwise zero loss has expected loss $0.02(50{,}000)=\$1{,}000$ per exposure. That does not predict a literal $1,000 loss on one occurrence. It describes repeated-exposure average modeled cost.

Mean and variance also compress information: distributions with the same summaries can have different shapes. Always inspect the model and business consequences.

## Executable NumPy audit and simulation

The implementation directly mirrors the definitions:

```python
mu = np.sum(values * probabilities)
variance = np.sum(((values - mu) ** 2) * probabilities)
std = np.sqrt(variance)
```

It creates a pandas contribution table with `outcome`, `probability`, `weighted_value`, `deviation`, `squared_deviation`, and `weighted_squared_deviation`. Simulation uses a fixed generator through the reusable PMF sampler:

```python
rng = np.random.default_rng(seed)
samples = rng.choice(values, size=10_000, p=probabilities)
samples.mean()
samples.var(ddof=0)
```

Run 10, 100, 1,000, and 10,000 trials. The cumulative mean moves around theoretical $E[X]$ rather than converging monotonically. For draws from a known model, `np.var(samples, ddof=0)` provides the population-style empirical variance of those realized values. Some pandas/NumPy sample-statistic conventions use `ddof=1`; that estimator distinction comes later.

Theoretical/model quantities are $E[X]=\mu$ and $Var(X)=\sigma^2$. Observed sample quantities include $\bar{x}$ and empirical variance. Later statistics chapters use sample quantities to learn about unknown model quantities.

```bash
python3 -m analytics_foundations chapter-18
```

The command produces five figures: weighted balance, same mean/different spread, profit deviations, simulated cumulative mean, and simulated empirical variance.

## Common misconceptions

1. **“Expectation predicts the next result.”** It is a weighted center, not a next-trial forecast.
2. **“Expectation must be possible.”** A fair die has expectation 3.5.
3. **“Variance is average distance.”** It is average *squared* deviation.
4. **“SD has squared units.”** Variance does; SD returns to original units.
5. **“$E[X^2]=(E[X])^2$.”** Their difference is generally positive variance.
6. **“Equal expected values make choices equivalent.”** Variability, downside, shape, and consequences differ.
7. **“Higher expected profit is always better.”** Liquidity, constraints, risk tolerance, objectives, and model uncertainty matter.

## Mastery checkpoints

### Concept

Explain what $E[X]$ means; why it is weighted; why it can be impossible; why signed deviations fail; why variance squares; why SD restores units; and how theoretical expectation differs from a sample mean.

### Arithmetic

For $P(X=-10)=.20$, $P(X=0)=.50$, $P(X=20)=.30$:

$$E[X]=(-10)(.20)+0(.50)+20(.30)=4.$$

Then

$$Var(X)=(-14)^2(.20)+(-4)^2(.50)+(16)^2(.30)=39.2+8+76.8=124,$$
$$SD(X)=\sqrt{124}\approx11.14.$$

### Transformation

Given $Y=4X-7$, $E[X]=10$, and $Var(X)=9$, calculate $E[Y]=33$, $Var(Y)=144$, and $SD(Y)=12$.

### Business

Option A has $E[A]=\$1,000$, $SD(A)=\$200$; Option B has $E[B]=\$1,150$, $SD(B)=\$2,500$. Which should be chosen? The statistics alone are insufficient: examine tolerance, downside consequences, liquidity, repetition, assumptions, and objectives.

### Execution

Implement expectation and squared-deviation variance from a PMF; verify the shortcut and SD; simulate; compare theoretical and empirical center/spread; compare equal-mean distributions; and verify transformation rules.

### Interpretation

A promotion's $800 expected profit does not mean the next promotion makes about $800; it is a probability-weighted repeated-outcome center. Two promotions with expectation $800 but SDs $100 and $2,000 are not economically equivalent.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** expectation and variance underpin distributions, sampling distributions, estimators, intervals, tests, ANOVA, and regression.
- **BUAD 512B — Business Modeling with Python:** NumPy makes weighted calculations, simulation, and risk comparisons executable.
- **BUAD 5112 — Competing Through Business Analytics:** decisions under uncertainty balance expected outcomes, variability, and downside exposure rather than optimize one average.

This independent preparation chapter does not reproduce any William & Mary course.
