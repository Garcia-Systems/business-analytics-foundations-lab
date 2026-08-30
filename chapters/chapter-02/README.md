# Chapter 2 — Exponents, Logs & Growth

> **Central outcome:** Linear change repeatedly adds. Exponential change repeatedly multiplies. Logarithms let us work backward through exponential relationships.

Chapter 1 used functions to model relationships. Here the question is: **what happens when a business quantity changes by a percentage rather than a fixed amount?** Our fictional subscription business, **James River Analytics**, begins with 500 customers.

## Learning objectives

By the end, you should be able to distinguish additive from multiplicative change; read positive, zero, negative, and fractional exponents; convert rates to growth factors; evaluate exponential models; use logarithms to solve growth questions and find doubling time; distinguish percent from percentage points; implement models with Python and NumPy; interpret their graphs; and challenge unrealistic long-range forecasts.

## 1. Repeated addition or repeated multiplication?

Compare two hypotheses:

\[
N_{\text{linear}}(t)=500+50t
\qquad
N_{\text{exp}}(t)=500(1.08)^t.
\]

Here \(N(t)\) is customers after \(t\) months and 500 is the initial value \(N_0\). The linear recurrence is

\[
N_{t+1}=N_t+50,
\]

so it repeatedly **adds** the same amount. Eight percent written as a decimal is 0.08, and its growth factor is

\[
1+0.08=1.08.
\]

Thus the exponential recurrence is

\[
N_{t+1}=1.08N_t,
\]

so it repeatedly **multiplies** by the same factor. Adding 8 customers and growing by 8% are not interchangeable: 8% of 500 is 40, and 8% of a later, larger base is more than 40.

## 2. Exponents record repeated multiplication

Start with a concrete meaning, not a rule:

\[
2^3=2\times2\times2=8.
\]

Likewise, three months of 8% growth means

\[
\begin{aligned}
500(1.08)^1 &= 500(1.08)=540,\\
500(1.08)^2 &= 540(1.08)=583.20,\\
500(1.08)^3 &= 583.20(1.08)=629.856.
\end{aligned}
\]

The exponent counts growth periods. The rate stays 8%, but the absolute additions are 40, 43.20, then 46.656 customers because each percentage is applied to a growing base.

### Extending the meaning

For nonzero \(a\), the definitions remain consistent with multiplication:

- \(a^1=a\): one copy of the base.
- \(a^0=1\): because \(a^1/a^1=a^{1-1}=a^0=1\).
- \(a^{-1}=1/a\): stepping an exponent down divides by \(a\). For example, \(2^{-1}=1/2\).
- \(a^{1/2}=\sqrt a\): it is the number which, multiplied by itself, gives \(a\). Thus \(2^{1/2}=\sqrt2\approx1.414\).
- More generally, \(a^{m/n}=(\sqrt[n]{a})^m\) when the real-valued root exists.

A fractional period can therefore be meaningful when the model treats time continuously between measurement dates. These meanings make \((1+r)^t\) readable without turning growth analysis into abstract rule practice.

### Four useful exponent relationships

Repeated factors explain the relationships:

\[
a^ma^n=a^{m+n},\qquad \frac{a^m}{a^n}=a^{m-n},\qquad
(a^m)^n=a^{mn},\qquad a^{-n}=\frac1{a^n}.
\]

Multiplying joins two groups of factors; dividing cancels factors; raising a power repeats its whole group; a negative exponent reverses multiplication. These are descriptions of repeated multiplication, not arbitrary formulas.

## 3. The discrete compound-growth model

The general model is

\[
N(t)=N_0(1+r)^t,
\]

where \(N_0\) is the starting quantity, \(r\) is the rate per period as a decimal, \(1+r\) is the growth factor, and \(t\) is the number of matching periods. A 5% rate gives factor 1.05; factor 1.12 implies 12% growth; factor 0.97 implies a 3% decline.

Before reading the complete table, predict when exponential growth will overtake linear growth. At first, 8% of 500 is only 40, less than the linear addition of 50.

| Month \(t\) | Linear \(500+50t\) | Exponential \(500(1.08)^t\) |
|---:|---:|---:|
| 0 | 500 | 500.00 |
| 1 | 550 | 540.00 |
| 2 | 600 | 583.20 |
| 3 | 650 | 629.86 |
| 6 | 800 | 793.44 |
| 12 | 1,100 | 1,259.09 |
| 24 | 1,700 | 3,170.59 |

The exponential model first exceeds the linear model at month 7. It curves upward because the same percentage creates a larger absolute change on each larger base.

### Compound growth travels across contexts

The model for money has identical structure:

\[
V(t)=V_0(1+r)^t.
\]

For example, $1,000 compounded at 5% annually becomes \(1000(1.05)^2=1102.50\) after two years. The same function can represent customers, revenue, investment value, costs, demand, or users. Mathematics describes the relationship independently of the label.

## 4. Percentage change is not percentage-point change

Relative percentage change is

\[
\text{percentage change}=\frac{\text{new}-\text{old}}{\text{old}}\times100\%.
\]

If monthly recurring revenue rises from $40,000 to $46,000, the change is

\[
\frac{46{,}000-40{,}000}{40{,}000}\times100\%=15\%.
\]

When a conversion rate moves from 20% to 25%, it rises **5 percentage points**, but its relative increase is

\[
\frac{25-20}{20}=0.25=25\%.
\]

“Percent” compares a change with its starting value; “percentage points” subtracts two rates. Business claims often become misleading when these are confused.

## 5. Logarithms work backward through growth

Suppose James River Analytics wants to know when the model reaches 1,000 customers:

\[
500(1.08)^t=1000 \quad\Longrightarrow\quad (1.08)^t=2.
\]

Now the unknown is an exponent. A logarithm answers **what exponent produced this number?**

\[
\log_b(x)=y \quad\text{means exactly the same as}\quad b^y=x.
\]

The natural logarithm, \(\ln(x)\), is the logarithm with base \(e\). Its deeper role can wait; Python and analytics libraries commonly provide it as `np.log`.

Taking natural logs of both sides and using \(\ln(a^t)=t\ln(a)\):

\[
\ln((1.08)^t)=\ln2,
\]
\[
t\ln(1.08)=\ln2,
\]
\[
t=\frac{\ln2}{\ln(1.08)}\approx9.01\text{ months}.
\]

This yields the compound doubling-time formula

\[
t_{\text{double}}=\frac{\ln2}{\ln(1+r)},
\]

where \(r\) is the rate per period and the answer uses those same periods. A result of 9.01 means the continuous-time equation reaches exactly double shortly after nine months; measurements at whole month-ends first exceed double at month 10.

## 6. Equation ↔ code

The teaching code deliberately mirrors the mathematics:

```python
import numpy as np


def linear_customers(months):
    return 500 + 50 * months


def exponential_customers(months):
    return 500 * np.power(1.08, months)

months = np.array([0, 1, 2, 3, 6, 12, 24])
linear = linear_customers(months)
exponential = exponential_customers(months)
doubling = np.log(2) / np.log(1.08)
```

Python's `1.08 ** months` and NumPy's `np.power(1.08, months)` both correspond to \((1.08)^t\). NumPy evaluates every period in an array without a manual loop. `np.log` computes \(\ln\), letting code work backward from an outcome to time.

## 7. Executable experiment and visual evidence

From the repository root, run:

```bash
python3 -m analytics_foundations chapter-02
```

The experiment prints selected values, verifies the 8% monthly change, calculates doubling time and crossover, and generates:

1. `figures/chapter-02-linear-vs-exponential.png` — equal additions form a line while compounded percentage growth curves upward.
2. `figures/chapter-02-repeated-multiplication.png` — customer totals and month-to-month absolute additions show why a constant rate does not mean a constant addition.

The 24-month horizon reveals the shape difference without relying on an absurd scale.

## 8. Business interpretation and model risk

> An 8% growth assumption does not mean adding the same number of customers every month.

The absolute increase grows because the percentage is applied to an increasingly large base. That mathematical consistency can become dangerous in a long-range forecast. Constant exponential growth ignores market saturation, competition, churn, capacity, changing acquisition costs, pricing changes, and operational constraints. A useful analyst asks when assumptions may stop holding rather than merely extending a curve.

> **Mathematical consistency does not guarantee business realism.**

## 9. Mastery checkpoints

### Concept checkpoint

1. What differs between adding 8 customers and growing by 8%?
2. What does \(t\) represent in \(N_0(1+r)^t\)?
3. Why must \(a^0=1\) for nonzero \(a\)?
4. What question does a logarithm answer?
5. Why does exponential growth curve upward while its rate stays constant?

### Hand-calculation checkpoint

For \(R(t)=1000(1.05)^t\), calculate \(R(0)\), \(R(1)\), \(R(2)\), the percentage growth per period, and approximate doubling time using \(\ln2/\ln1.05\). Show the first two multiplications before using software.

### Execution checkpoint

Change the customer growth rate and regenerate both figures. Compare 5% with 7%, calculate each doubling time, and compare the values after 10 and 20 periods. Explain why a two-point rate difference creates an increasingly large outcome difference.

### Interpretation checkpoint

Model A predicts 20% annual growth indefinitely; Model B predicts 8%. After 20 years A is dramatically larger. Does that prove A is the better forecast? No: arithmetic shows consequences **if assumptions hold**, not whether saturation, competition, capacity, churn, and other assumptions are credible. Identify evidence you would seek before choosing.

## 10. W&M preparation connection

This independent chapter does not reproduce any William & Mary course.

- **BUAD 512A — Probability & Statistics with R:** exponential and logarithmic relationships recur in distributions, transformations, statistical models, and likelihood methods.
- **BUAD 512B — Business Modeling with Python:** growth relationships can be expressed mathematically and evaluated efficiently with Python and NumPy.
- **BUAD 5112 — Competing Through Business Analytics:** rates, compound change, metrics, and forecast assumptions require careful business interpretation.

Chapter 3 remains future work. First retain the core distinction: **linear change repeatedly adds; exponential change repeatedly multiplies; logarithms let us work backward.**
