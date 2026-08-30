# Chapter 4 — Change & Derivatives

> **Central outcome:** A derivative measures how quickly the output of a model changes as its input changes. It emerges from average rates of change over smaller and smaller intervals, and in business it gives us a language for marginal effects.

Previous chapters made relationships into functions, modeled multiplicative growth, and aggregated observations. Now ask: **If an input changes, how quickly does the output change?** We will move from change between two points to an instantaneous rate—not begin with rules.

## Learning objectives

By the end, you should be able to calculate absolute and average change; interpret secant and tangent slopes; explain shrinking intervals and derivative notation; approximate derivatives; differentiate simple polynomials; interpret business units and marginal revenue, cost, and profit; connect derivative signs and zero to model behavior; use Python and NumPy; and distinguish continuous models from discrete operations.

## 1. Ordinary change in a business model

Market research for fictional **Harbor Prints** suggests that selling more requires a lower price:

\[
p(q)=30-0.05q.
\]

Thus revenue, cost, and profit are

\[
R(q)=qp(q)=30q-0.05q^2,
\]
\[
C(q)=200+10q,
\]
\[
P(q)=R(q)-C(q)=20q-0.05q^2-200.
\]

Here $q$ is units sold and each output is dollars. The Greek capital delta means “change in”:

\[
\Delta y=y_2-y_1,\qquad \Delta x=x_2-x_1.
\]

At 100 units, $P(100)=20(100)-0.05(100)^2-200=1{,}300$. At 110, $P(110)=2{,}200-605-200=1{,}395$. Therefore

\[
\Delta P=1{,}395-1{,}300=95,\qquad \Delta q=110-100=10.
\]

Profit increased by **$95**; that is absolute output change, not a rate.

## 2. Average rate of change

The average rate is

\[
\frac{\Delta y}{\Delta x}=\frac{f(x_2)-f(x_1)}{x_2-x_1}.
\]

$f$ is the relationship; $x_1,x_2$ are initial and final inputs; $f(x_1),f(x_2)$ are their outputs; the numerator is output change; the denominator is input change. For Harbor Prints,

\[
\frac{P(110)-P(100)}{110-100}=\frac{95}{10}=9.50
\quad\frac{\text{dollars of profit}}{\text{unit sold}}.
\]

So profit changes by **$9.50 per additional unit on average** across 100–110. The units of any rate are output units divided by input units.

A wider interval gives a different answer. $P(120)=1{,}480$, so the average from 100 to 120 is $(1480-1300)/20=9$ dollars per unit. Over the narrower 100–105 interval, $P(105)=1{,}348.75$, giving $48.75/5=9.75$ dollars per unit. Nonlinear functions do not have one constant slope.

## 3. The secant line

A line through $(q_1,P(q_1))$ and $(q_2,P(q_2))$ is a **secant line**. Its slope is the average rate of change across that interval. Run the experiment to generate `chapter-04-average-rate.png`, where the two marked points, their horizontal interval, and secant make the calculation geometric.

## 4. Shrinking the interval

Hold $q=100$ fixed and move the second point to $100+h$:

| $h$ (units) | $[P(100+h)-P(100)]/h$ ($/unit) |
|---:|---:|
| 20 | 9.0000 |
| 10 | 9.5000 |
| 5 | 9.7500 |
| 1 | 9.9500 |
| 0.1 | 9.9950 |
| 0.01 | 9.9995 |

The values approach 10. This numerical pattern comes before formal notation: the second point moves closer, the secant interval shrinks, and its slope approaches the slope right at 100.

## 5. The derivative emerges

We name that instantaneous rate $P'(100)=10$ dollars of profit per unit. In general,

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
\]

- $f(x)$ is the original output.
- $f(x+h)$ is output after input changes by $h$.
- The numerator is output change.
- The quotient is an average rate across an interval of width $h$.
- $h\to0$ asks what happens as that interval becomes arbitrarily small.
- The limit is the value these rates approach.

**Do not set $h=0$.** That would produce $[f(x+0)-f(x)]/0=0/0$, division by zero. A limit asks what value the quotient approaches while $h$ is nonzero but closer and closer to zero.

The tangent line touches the curve locally and represents instantaneous change. `chapter-04-secant-to-tangent.png` contrasts one wide secant, one narrow secant, and the tangent. **Secant slope measures change across an interval; tangent slope represents instantaneous change at a point.**

The notations $f'(x)$ and $df/dx$ both mean derivative with respect to $x$. Correspondingly, $R'(q)$, $C'(q)$, and $P'(q)$ are rates with respect to quantity. Their units here are dollars per unit sold.

## 6. Minimal symbolic tools—after the idea

Only four rules are needed:

\[
\frac{d}{dx}c=0,\quad \frac{d}{dx}x^n=nx^{n-1},
\]
\[
\frac{d}{dx}[cf(x)]=cf'(x),\quad
\frac{d}{dx}[f(x)\pm g(x)]=f'(x)\pm g'(x).
\]

Apply them term by term:

\[
R'(q)=\frac{d}{dq}[30q-0.05q^2]=30(1)-0.05(2q)=30-0.1q,
\]
\[
C'(q)=\frac{d}{dq}[200+10q]=0+10=10,
\]
\[
P'(q)=R'(q)-C'(q)=(30-0.1q)-10=20-0.1q.
\]

At $q=150$, $P'(150)=20-15=5$ dollars of profit per unit. This is **marginal profit**, not total profit. In fact $P(150)=1{,}675$.

## 7. Marginal business meaning and discrete reality

$R'(q)$ is marginal revenue, $C'(q)$ marginal cost, and $P'(q)$ marginal profit: continuous-model rates near $q$. At 100 they are respectively $20, $10, and $10 per unit, and $P'=R'-C'$.

Real products may come only in whole units. The actual modeled one-unit change is

\[
P(q+1)-P(q).
\]

At 100 this is $P(101)-P(100)=9.95$ dollars, close to but not identical to $P'(100)=10$ dollars per unit. A derivative is a local continuous approximation, not a promise about an indivisible next sale.

## 8. Positive, zero, and negative

- $P'(q)>0$: profit is locally increasing.
- $P'(q)=0$: profit is locally flat.
- $P'(q)<0$: profit is locally decreasing.

Solve

\[
20-0.1q=0\implies q=200.
\]

The downward-opening profit curve reaches its top there; `chapter-04-profit-and-marginal-profit.png` aligns the profit maximum with the zero crossing below it. $P(200)=1{,}800$, so **marginal profit = 0 does not mean total profit = 0**. Profit is a level, change in profit compares levels, and marginal profit is a rate. A derivative can tell us where increasing an input stops improving an output, without yet creating a complete optimization course.

## 9. Numerical derivatives in Python

```python
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)
```

Forward difference mirrors the shrinking table. Central difference uses points equally far to both sides; geometrically its secant is balanced around $x$ and often gives a closer local estimate. For this quadratic, at $q=100,h=0.01$, forward difference is 9.9995, central difference is 10, and the analytical derivative is 10. NumPy creates quantity grids and evaluates the same functions across every value for the figures.

```python
def marginal_revenue(q):
    return 30 - 0.1 * q

def marginal_cost(q):
    return 10

def marginal_profit(q):
    return 20 - 0.1 * q
```

## 10. Assumptions and management limits

The mathematical optimum is not automatically the operating plan. The model treats quantity as continuous and assumes certain demand, sufficient capacity and inventory, available labor, no competitor response, and unchanged costs. Actual quantities are discrete and forecasts uncertain. Management should test feasibility and sensitivity before acting.

## Mastery checkpoints

### Concept

1. What does $\Delta y/\Delta x$ represent, and what are its units?
2. What is a secant line? What changes when its interval shrinks?
3. Why can we not substitute $h=0$ into the difference quotient?
4. What does $P'(100)$ mean and what are its units?
5. What does $P'(q)=0$ tell us—and what does it not tell us?

### Hand calculation

For $P(q)=24q-0.06q^2-250$:

1. Calculate $P(50)$ and $P(60)$.
2. Calculate average rate of change from 50 to 60.
3. Derive $P'(q)$ using the power rule.
4. Calculate $P'(50)$.
5. Solve $P'(q)=0$.

Check after working: $P(50)=800$, $P(60)=974$, average $=17.4$/unit, $P'(q)=24-0.12q$, $P'(50)=18$/unit, and the zero occurs at $q=200$.

### Execution

Change price, variable cost, or fixed cost parameters. Predict the curves first, regenerate all figures, compare forward and central differences, and try several $h$ values. What happens for extremely tiny $h$? Floating-point rounding means computationally “smaller” is not always “better”; no deeper precision analysis is needed here.

### Interpretation

At 150 units, marginal profit is $5 per unit. Does that mean total profit is $5? **No:** it describes local change; total profit is $1,675. The model says maximum profit occurs at 200 units. Should management automatically produce exactly 200? **No:** demand uncertainty, capacity, inventory, labor, competitors, discrete units, and changing costs may alter what is feasible or profitable.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** rates and optimization support later understanding of densities, statistical models, estimation, and model behavior.
- **BUAD 512B — Business Modeling with Python:** functions can represent business relationships and be investigated analytically and numerically.
- **BUAD 5112 — Competing Through Business Analytics:** marginal reasoning distinguishes total performance from how it changes with a management decision.

This independent preparation chapter does not reproduce any William & Mary course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-04
```

Inspect the table and three figures. Confirm that numerical rates approach 10 at $q=100$, the tangent differs from secants, and the profit maximum at 200 aligns with zero marginal profit.
