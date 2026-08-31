# Chapter 19 — Covariance & Dependence

> **Central question:** When two business quantities change together, how can we measure the direction and strength of that relationship?

```text
customers: 120, 150, 180, 210, ...
revenue:  2200, 2700, 3300, 3900, ...
```

Are these two lists related? Separate lists lose the essential structure. The observations are pairs
\((x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\), and both entries in a pair must describe the same case. Here **one pair is customer count and revenue for the same location-day**. This grain discipline carries forward from Chapter 12: sorting or joining only one column would destroy the question.

## See the pairs before summarizing them

Start with scatterplots of customers–revenue, labor-hours–revenue, and temperature–revenue. As in Chapter 13, describe **direction, form, strength, and unusual points** before calculating. Customers and revenue have a clear positive, roughly linear tendency; labor and revenue move together; temperature is comparatively weak. Can we turn the visual tendency into a numerical measure?

## Products of paired deviations

Chapter 18 used \(x_i-\bar x\) to say whether one value is above or below its mean. For the same observation calculate both \(x_i-\bar x\) and \(y_i-\bar y\), then multiply:

\[(x_i-\bar x)(y_i-\bar y).\]

The sign logic is the heart of covariance:

* both above: \((+)(+)=+\), a positive contribution;
* both below: \((-)(-)=+\), also positive;
* opposite sides: \((+)(-)=-\), a negative contribution.

Thus positive covariance comes from variables usually occupying the same side of their means; negative covariance comes from opposite sides. The paired-deviations figure draws the two mean lines: upper-right and lower-left points contribute positively, while upper-left and lower-right points contribute negatively.

## Example 1 — arithmetic exposed

| Day | \(x_i\) customers | \(y_i\) revenue | \(x_i-\bar x\) | \(y_i-\bar y\) | Product |
|---:|---:|---:|---:|---:|---:|
|1|100|2,000|-75|-975|73,125|
|2|150|2,600|-25|-375|9,375|
|3|200|3,400|25|425|10,625|
|4|250|3,900|75|925|69,375|
|**sum**|||||**162,500**|

Here \(\bar x=175\), \(\bar y=2,975\), and the sample covariance is \(162,500/(4-1)=54,166.67\) customer-dollars. The many same-side pairs make it positive. For a negative example, \(x=[1,2,3]\), \(y=[3,2,1]\) has products \(-1,0,-1\), so sample covariance is \(-1\). A random cloud may balance near zero; a U-shape can also balance despite strong dependence.

## Population and sample covariance

For a probability model,

\[\operatorname{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)].\]

Compare \(\operatorname{Var}(X)=E[(X-\mu_X)^2]\): **variance is covariance with itself**, \(\operatorname{Cov}(X,X)=\operatorname{Var}(X)\). For observed pairs, the standard sample convention is

\[s_{XY}=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y).\]

We defer the deeper reason for \(n-1\) to Part IV. Positive covariance indicates same-direction linear co-movement, negative indicates opposite-direction co-movement, and near zero indicates little *linear* co-movement—not necessarily independence.

## The units problem motivates correlation

Customer–revenue covariance has awkward units customers × dollars. Let \(Y_{cents}=100Y_{dollars}\). Then

\[\operatorname{Cov}(X,Y_{cents})=100\operatorname{Cov}(X,Y_{dollars}),\]

although nothing substantive changed. A large covariance may merely reflect large units. How can it become scale-free?

Pearson correlation standardizes covariance:

\[\rho_{XY}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y},\qquad r=\frac{s_{XY}}{s_Xs_Y}.\]

It is unitless and \(-1\le r\le1\). Positive and negative values give linear direction, \(r\approx0\) means weak linear association, and \(|r|\approx1\) means points lie close to a line. Strength is contextual; there is no universal “0.7 is strong” rule. Dollars-to-cents leaves \(r\) unchanged. Any positive rescaling (or positive affine temperature conversion) preserves it; negative scaling reverses its sign.

For \(Y=2X+10\), \(r=1\); for \(Y=100-3X\), \(r=-1\). Yet \(Y=2X\) and \(Z=200X\) both have correlation 1 despite dramatically different slopes. **Correlation is standardized linear association, not rate of change.** Regression, introduced later, estimates slopes.

Chapter 17's standardized scores make this intuitive: \(z_x=(x-\bar x)/s_x\), \(z_y=(y-\bar y)/s_y\). Correlation is closely related to averaging \(z_xz_y\): do standardized deviations share sign and magnitude?

## Python and the matrix connection

```python
x = df["customers"].to_numpy()
y = df["revenue"].to_numpy()
cov_matrix = np.cov(x, y, ddof=1)
corr = np.corrcoef(x, y)[0, 1]
df[["customers", "revenue", "labor_hours"]].corr()
```

NumPy's two-variable covariance result is

\[\begin{bmatrix}\operatorname{Var}(X)&\operatorname{Cov}(X,Y)\\\operatorname{Cov}(Y,X)&\operatorname{Var}(Y)\end{bmatrix}.\]

More generally, \(\Sigma\) puts variances on its diagonal and pairwise covariances off diagonal. It is symmetric. This connects directly to Chapter 7 matrices. A pandas Pearson correlation matrix is likewise symmetric with ones on its diagonal. The experiment uses Matplotlib `imshow`, not seaborn, to annotate customers, revenue, labor hours, and temperature. The transparent helpers validate equal lengths, at least two finite observations, and reject correlation with zero variability.

## Correlation tells what, not why

A positive \(\operatorname{Corr}(labor,revenue)\) does **not** justify “adding labor causes revenue.” Managers may schedule labor because they expect demand:

\[Expected\ Demand\rightarrow Labor,\qquad Expected\ Demand\rightarrow Revenue.\]

Labor could affect service and revenue, but an association alone cannot distinguish that story. Weekends, promotions, reservations, location size, season, and local events can confound it. Similarly, temperature may drive both ice-cream and cold-drink sales. Correlation tells what moves together, not why; it never establishes causation by itself.

## Influential points and restricted ranges

Pearson correlation is sensitive to unusual observations. The outlier figure uses the same weak cloud with and without one extreme point and reports both correlations. Inspect points, rather than trusting one summary. Likewise, observing only a narrow customer range can reduce correlation even when the broader population follows a strong tendency; summaries depend on the range observed.

## Nonlinear dependence: the memorable counterexample

Let \(X\in\{-3,-2,-1,0,1,2,3\}\) and \(Y=X^2\). Knowing X determines Y perfectly, yet symmetry makes Pearson \(r=0\): the U-shape has no overall linear direction. **Zero correlation means no linear association measured by Pearson correlation, not no relationship.**

Two variables are dependent when information about one changes the probability model for the other. Independence would require \(P(X\in A\mid Y\in B)=P(X\in A)\) for appropriate events. Independence with suitable finite moments implies zero covariance, but the reverse fails, as \(Y=X^2\) shows. Covariance and correlation summarize only part of joint behavior; full joint distributions are beyond this chapter.

## Why covariance matters for combined uncertainty

\[\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).\]

If the variances are 100 and 225: covariance 50 gives \(100+225+2(50)=425\); covariance zero gives 325; covariance −50 gives 225. Positively moving product revenues amplify combined volatility, while weak or negative co-movement can offset it—simple diversification intuition, not portfolio theory. **Advanced recognition preview:** for weights \(w\) and covariance matrix \(\Sigma\), \(\operatorname{Var}(w^TX)=w^T\Sigma w\). No derivation or mastery is expected here.

## Common misconceptions

1. **Positive covariance proves causation.** No: it measures co-movement.
2. **Correlation 0 means unrelated.** No: it means no Pearson linear association.
3. **Correlation 1 means equal variables.** No: points occupy a perfectly increasing line.
4. **Correlation gives slope.** No: it is standardized and unitless.
5. **Large covariance means strong.** Not necessarily: covariance depends on units.
6. **Zero covariance means independence.** Independence generally implies zero covariance (with suitable moments), not conversely.
7. **Outliers cannot distort correlation.** Pearson correlation can be highly sensitive to influential points.

## Mastery checkpoints

### Concept
Explain covariance, deviation-product signs, awkward units, correlation's scale solution, what \(r=0\) means, why correlation is neither slope nor causation, and why \(\operatorname{Cov}(X,X)=\operatorname{Var}(X)\).

### Hand calculation
For \(x=[1,2,3]\), \(y=[2,4,5]\), produce an intermediate table containing both means, both deviations, and products; then calculate sample covariance, sample standard deviations, and correlation. Do not skip the table.

### Sign, scale, and dependence
Predict before computing: variables rising together; one rising while another falls; a U-shape; a random cloud. If \(Y^*=100Y\), what happens to covariance and correlation? (Covariance ×100; correlation unchanged.) If \(Corr(X,Y)=0\), can Y depend on X? (Yes: \(Y=X^2\).)

### Business interpretation
Labor hours and revenue have correlation 0.82. Should management add labor to raise revenue? **Not from correlation alone:** expected demand may drive both, with other confounders present.

### Execution
Run the experiment; manually calculate and verify covariance with NumPy; calculate correlation; make a scatterplot; change units; inspect covariance and correlation matrices; construct nonlinear zero-correlation dependence; and calculate variance of a sum.

```bash
python3 -m analytics_foundations chapter-19
```

## W&M preparation connection

* **BUAD 512A — Probability & Statistics with R:** covariance and correlation underpin multivariable statistics, regression, ANOVA-related reasoning, and dependence.
* **BUAD 512B — Business Modeling with Python:** NumPy and pandas execute paired calculations and relationship diagnostics.
* **BUAD 5112 — Competing Through Business Analytics:** interpretation must distinguish association, shared drivers, and causal claims.

This independent preparation chapter does not reproduce any W&M course.

> **Outcome:** Covariance measures whether two variables tend to move together, while correlation expresses that linear co-movement on a standardized scale. These measures can miss nonlinear relationships, be influenced by unusual observations, and never establish causation by themselves.
