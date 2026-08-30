# Chapter 8 — Linear Algebra for Models

> **Central outcome:** Linear algebra lets us express models as systems of weighted relationships. Exact systems can sometimes be solved directly, while real analytical data often requires choosing coefficients that best approximate many noisy observations.

Chapter 6 introduced vectors and Chapter 7 introduced matrices. We now use them to answer four business questions: How can several constraints be solved simultaneously? When is a solution unique? What happens when a feature adds no information? Why does regression seek the *best* coefficients rather than an exact fit?

## Learning objectives

By the end, you can form linear combinations; translate equations into and read $A\mathbf{x}=\mathbf b$; solve and verify a small system by hand and with NumPy; interpret inverses, span, dependence, and rank; distinguish unique, dependent, and inconsistent systems; read $X\boldsymbol\beta$ as predictions; and calculate least-squares residuals. This is the useful foundation for analytics and regression—not a complete linear algebra course.

## Linear combinations: columns are building blocks

Let

\[
\mathbf v_1=\begin{bmatrix}1\\2\end{bmatrix},\qquad
\mathbf v_2=\begin{bmatrix}3\\1\end{bmatrix}.
\]

A **linear combination** multiplies vectors by scalars, then adds the results. For example,

\[
2\mathbf v_1+3\mathbf v_2
=\begin{bmatrix}2\\4\end{bmatrix}+\begin{bmatrix}9\\3\end{bmatrix}
=\begin{bmatrix}11\\7\end{bmatrix}.
\]

Place those vectors in the columns of $A$, and their weights in \(\mathbf x\):

\[
A=\begin{bmatrix}1&3\\2&1\end{bmatrix},\quad
\mathbf x=\begin{bmatrix}a\\b\end{bmatrix},\quad
A\mathbf x=a\begin{bmatrix}1\\2\end{bmatrix}+b\begin{bmatrix}3\\1\end{bmatrix}.
\]

Thus matrix-vector multiplication is not mysterious: it combines the columns of $A$ using the entries of the coefficient vector. With $a=2,b=3$, it produces $[11,7]^T$, exactly as above.

## A two-package contribution system

A fictional service firm observes two clean days. Standard and premium packages have unknown, constant contributions $s$ and $p$:

\[
\begin{aligned}
2s+p&=110 &&\text{(day 1)}\\
s+3p&=170 &&\text{(day 2)}.
\end{aligned}
\]

In compact form,

\[
\underbrace{\begin{bmatrix}2&1\\1&3\end{bmatrix}}_A
\underbrace{\begin{bmatrix}s\\p\end{bmatrix}}_{\mathbf x}
=
\underbrace{\begin{bmatrix}110\\170\end{bmatrix}}_{\mathbf b}.
\]

Here $A$ contains observed package quantities, \(\mathbf x\) contains unknown per-package contributions, and \(\mathbf b\) contains observed daily totals. The matrix equation is simply both business equations packaged together.

### Solve by elimination, then verify

Multiply the second equation by 2: $2s+6p=340$. Subtract the first equation: $5p=230$, so $p=46$. Substitution gives $s+3(46)=170$, hence $s=32$. Verification matters:

\[
2(32)+46=110,\qquad 32+3(46)=170.
\]

The coefficient vector $[32,46]^T$ is both the intersection of two lines and the solution of the business system.

```python
import numpy as np
A = np.array([[2., 1.], [1., 3.]])
b = np.array([110., 170.])
x = np.linalg.solve(A, b)
np.testing.assert_allclose(A @ x, b)
```

If an inverse $A^{-1}$ exists, $A^{-1}A=I$, so conceptually \(\mathbf x=A^{-1}\mathbf b\). But use `np.linalg.solve(A, b)`, not `np.linalg.inv(A) @ b`: direct solving states the goal more clearly and is generally more numerically stable and efficient. The inverse is useful intuition, not a required intermediate calculation.

## One, many, or no exact solutions

For two unknowns, each equation draws a line.

- **Unique:** independent lines meet once, as the package equations do. There is one coefficient vector.
- **Dependent:** $x+2y=6$ and $2x+4y=12$ are the same line. The second constraint adds nothing, so infinitely many points work.
- **Inconsistent:** $x+2y=6$ and $2x+4y=15$ are parallel distinct lines. No point satisfies both.

One vector is **linearly dependent** on others if it can be constructed from them. If \(\mathbf v_2=2\mathbf v_1\), it supplies no new direction. This is the geometry behind the dependent system.

## Span, column space, and rank—practical meanings

The **span** of vectors is every result obtainable from their linear combinations. In 2D, one nonzero vector spans a line; two independent vectors span the plane. An exact solution to $A\mathbf x=\mathbf b$ exists only when \(\mathbf b\) lies in the span of $A$'s columns, gently called its **column space**.

**Rank** counts the independent directions, or practical information dimensions, represented by a matrix:

\[
\operatorname{rank}\begin{bmatrix}1&0\\0&1\end{bmatrix}=2,\qquad
\operatorname{rank}\begin{bmatrix}1&2\\2&4\end{bmatrix}=1.
\]

The second matrix has rank 1 because column 2 is twice column 1. NumPy checks this with `np.linalg.matrix_rank(A)`. To distinguish dependence from inconsistency, compare the rank of $A$ with the augmented matrix $[A\mid\mathbf b]$: a larger augmented rank means the total cannot be represented by the original columns.

A business dataset containing `revenue` and `revenue_in_cents = 100 * revenue` illustrates redundancy. The scaled column adds essentially no analytical information. Redundant model features can prevent coefficients from being uniquely identified even though they make the table wider.

## Features and model coefficients

The Chapter 7 preview now has a meaning:

\[
\hat{\mathbf y}=X\boldsymbol\beta.
\]

Rows of $X$ are observations, columns are features, \(\boldsymbol\beta\) contains unknown feature coefficients, and \(\hat{\mathbf y}\) contains predictions. Each prediction is the row-wise linear combination

\[
\hat y_i=\beta_1x_{i1}+\beta_2x_{i2}+\cdots.
\]

For a line \(\hat y=\beta_0+\beta_1x\), a column of ones carries the intercept:

\[
X=\begin{bmatrix}1&x_1\\1&x_2\\\vdots&\vdots\\1&x_n\end{bmatrix},\quad
\boldsymbol\beta=\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}.
\]

Then $X\boldsymbol\beta$ produces every prediction at once.

## More observations than unknowns

Suppose five days give $X\in\mathbb R^{5\times2}$, $\boldsymbol\beta\in\mathbb R^2$, and $\mathbf y\in\mathbb R^5$. Five noisy equations constrain only two coefficients. Usually no single coefficient vector satisfies all five exactly.

For $x=[1,2,3,4,5]$ and observed $y=[9,12,16,19,23]$, use an intercept and slope. NumPy finds approximately \(\hat\beta_0=5.3\), \(\hat\beta_1=3.5\). The first prediction is $5.3+3.5(1)=8.8$, so its residual is

\[
e_1=y_1-\hat y_1=9-8.8=0.2.
\]

In general,

\[
\mathbf e=\mathbf y-X\boldsymbol\beta.
\]

**Least squares** answers “which coefficients?” by minimizing

\[
\sum_{i=1}^n e_i^2=\lVert\mathbf y-X\boldsymbol\beta\rVert^2.
\]

This brings together Chapter 3's summation, Chapter 6's vector magnitude, and Chapter 7's matrix multiplication.

```python
beta, reported_residuals, rank, singular_values = np.linalg.lstsq(
    X, y, rcond=None
)
predictions = X @ beta
residuals = y - predictions
rss = residuals @ residuals
```

As a recognition-only preview, least-squares coefficients are associated with $X^TX\boldsymbol\beta=X^T\mathbf y$, sometimes written \(\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf y\). We do not derive or memorize that here, and numerical software commonly avoids explicitly computing the inverse. Chapter 26 returns to regression properly.

## Business interpretation and limitations

Exact algebraic systems are useful teaching cases, but real observations vary because of measurement noise, omitted variables, changing customer behavior, price changes, promotions, operational variation, and model simplification. Analytical modeling therefore moves from solving exact equations to finding coefficients that best approximate many observations.

A tiny residual sum of squares does **not** establish causality, future predictive value, or strategic usefulness. Fit must be considered alongside data quality, validation, plausible mechanisms, costs, risks, and the decision context.

## Mastery checkpoints

### Concept checkpoint

1. What is a linear combination, and how does matrix-vector multiplication create one?
2. What does each object in $A\mathbf x=\mathbf b$ mean?
3. What does a unique solution represent? What does linear dependence mean?
4. Define span without formal vector-space language. What does rank measure intuitively?
5. Why might real data have no exact solution?
6. What is a residual? What quantity does least squares minimize?

### Hand-calculation checkpoint

Solve $2x+y=8$ and $x+2y=7$ by hand. Write its matrix form and verify your answer in both equations. Then explain why $x+2y=4$ and $2x+4y=8$ do not have a unique solution.

### Execution checkpoint

Create a small system, solve it with `np.linalg.solve`, and verify it with matrix multiplication. Modify one total to make a dependent-coefficient system inconsistent and inspect the coefficient and augmented ranks. Create an overdetermined system, solve with `np.linalg.lstsq`, and manually reconcile one prediction and residual.

### Interpretation checkpoint

- A model has a very small residual sum of squares. Does that prove it is useful for business decisions? Explain why numerical fit alone cannot establish causal validity, future accuracy, or business value.
- Two feature columns are perfectly dependent. What new information does the second contribute? **Essentially none.**

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** regression and multivariable methods use vectors and matrices; residuals, rank, and least squares make later formulas interpretable.
- **BUAD 512B — Business Modeling with Python:** NumPy linear algebra tools efficiently solve systems and fit numerical relationships.
- **BUAD 5112 — Competing Through Business Analytics:** a model can summarize relationships, but good fit does not automatically make it strategically useful.

This independent preparation chapter does not reproduce any William & Mary course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-08
```

It solves and verifies the package system, distinguishes dependent and inconsistent systems, reports ranks, fits noisy observations, reconciles residuals and RSS, and writes four figures: unique, dependent, inconsistent, and least-squares views.
