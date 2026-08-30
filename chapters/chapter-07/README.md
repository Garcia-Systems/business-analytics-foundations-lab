# Chapter 7 — Matrices: Data Becomes Structure

> **Central outcome:** A matrix is a structured collection of vectors. Rows can represent observations, columns can represent features, and matrix multiplication lets us apply weighted relationships across many observations at once.

Chapter 6 represented one customer as a vector. Now we ask: **How can an entire business dataset be represented mathematically? How can we calculate the same weighted score for every customer without repeating it? How do dimensions tell us whether an operation makes sense?**

## Learning objectives

By the end, you should be able to explain a matrix; read entries, rows, columns, and dimensions; connect a DataFrame to a matrix; distinguish row and column vectors; transpose, add, subtract, and scale matrices; perform and dimension-check matrix-vector and matrix-matrix multiplication; recognize multiplication as repeated dot products; distinguish `*` and `@`; and see why matrices prepare us for multivariable analytics and regression.

## 1. From customer vectors to a data matrix

Consider three fictional Harbor Cafe customers, each described in the order visits, average order value, and months as customer:

\[
\mathbf{x}_1=\begin{bmatrix}4\\25\\12\end{bmatrix},\quad
\mathbf{x}_2=\begin{bmatrix}7\\18\\6\end{bmatrix},\quad
\mathbf{x}_3=\begin{bmatrix}3\\40\\24\end{bmatrix}.
\]

Stack their transposes as rows:

\[
X=\begin{bmatrix}4&25&12\\7&18&6\\3&40&24\end{bmatrix}.
\]

A **matrix** is a rectangular arrangement of entries. Here rows are observations/customers and columns are features. The labeled pandas DataFrame, a NumPy 2D array, and the mathematical matrix are different representations of the same structured numerical data. The DataFrame retains labels and business meaning; the matrix exposes numerical structure. Analysts need both and should not discard labels casually.

A row $[4\;25\;12]$ is one observation as a **row vector**. A column $[4\;7\;3]^T$ is one feature—visits—across observations. Orientation changes meaning.

## 2. Entries and dimensions

In $x_{ij}$, $i$ identifies the row/observation and $j$ identifies the column/feature. Thus $x_{23}=6$: row 2 is Customer B and column 3 is tenure, so Customer B has been a customer for 6 months. Mathematical indices begin at 1; NumPy's equivalent is `X[1, 2]` because Python starts at 0.

The example has 3 observations and 3 features:

\[
X\in\mathbb R^{3\times3}.
\]

Generally, $X\in\mathbb R^{m\times n}$ means $m$ rows and $n$ columns. NumPy reports this as `X.shape == (m, n)`; `X.ndim == 2` confirms two axes. This is not the count of entries: an $m\times n$ matrix has $mn$ entries.

```python
X.shape       # (3, 3)
X.ndim        # 2
X[1, 2]       # row 2, column 3 in mathematical notation
X[0, :]       # first customer row
X[:, 0]       # visits column
```

## 3. Transpose: exchange rows and columns

Transpose swaps rows and columns:

\[
X^T=\begin{bmatrix}4&7&3\\25&18&40\\12&6&24\end{bmatrix}.
\]

If $X\in\mathbb R^{m\times n}$, then $X^T\in\mathbb R^{n\times m}$. NumPy uses `X.T`. Transpose matters because datasets commonly put observations in rows while equations sometimes require feature vectors in columns or features in rows. Later formulas will use $X^T$; no advanced transpose identities are needed here.

For a clearer nonsquare hand example:

\[
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
A^T=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}.
\]

The shape changes from $2\times3$ to $3\times2$.

## 4. Addition, subtraction, and scalar multiplication

Same-shaped matrices add and subtract entry by entry:

\[
\begin{bmatrix}1&2\\3&4\end{bmatrix}+
\begin{bmatrix}5&6\\7&8\end{bmatrix}=
\begin{bmatrix}6&8\\10&12\end{bmatrix}.
\]

Different shapes are incompatible because corresponding entries cannot be paired. Scalar multiplication applies one number to every entry:

\[
2\begin{bmatrix}1&2\\3&4\end{bmatrix}=
\begin{bmatrix}2&4\\6&8\end{bmatrix}.
\]

These operations are element-wise; multiplication below has a different structure.

## 5. The central experiment: one rule for every customer

Choose the deliberately illustrative weight vector

\[
\mathbf w=\begin{bmatrix}2\\0.1\\0.5\end{bmatrix}.
\]

For one customer, the score is the Chapter 6 dot product $s_i=\mathbf x_i\cdot\mathbf w$. Compute all three by hand:

\[
s_1=4(2)+25(0.1)+12(0.5)=16.5,
\]
\[
s_2=7(2)+18(0.1)+6(0.5)=18.8,
\]
\[
s_3=3(2)+40(0.1)+24(0.5)=22.0.
\]

Stacked together,

\[
X\mathbf w=\begin{bmatrix}16.5\\18.8\\22.0\end{bmatrix}=\mathbf s.
\]

> Matrix-vector multiplication is a collection of row-by-vector dot products.

`X @ weights` performs the same weighted calculation for every row without a loop. That is the chapter's major insight.

## 6. Why dimensions must match

If $A\in\mathbb R^{m\times n}$ and $B\in\mathbb R^{n\times p}$, then

\[
(m\times n)(n\times p)\longrightarrow(m\times p).
\]

The inside dimensions match and the outside dimensions describe the result. This is not merely a memory trick: every row of $A$ contains $n$ elements and every column of $B$ contains $n$ elements, so their dot product exists.

- $(3\times3)(3\times1)\to(3\times1)$: three customer scores.
- $(6\times3)(3\times2)\to(6\times2)$: two outputs for six customers.
- $(2\times3)(3\times4)\to(2\times4)$: compatible repeated dot products.
- $(6\times3)(2\times2)$ is incompatible: 3 and 2 do not match, so a three-entry row cannot be dotted with a two-entry column.

## 7. Matrix-matrix multiplication is repeated dot products

Let

\[
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}.
\]

For $C=AB$, each $c_{ij}$ is row $i$ of $A$ dotted with column $j$ of $B$:

\[
AB=\begin{bmatrix}
1(5)+2(7)&1(6)+2(8)\\
3(5)+4(7)&3(6)+4(8)
\end{bmatrix}
=\begin{bmatrix}19&22\\43&50\end{bmatrix}.
\]

NumPy expresses this as `A @ B` or `np.matmul(A, B)`.

### Element-wise is not matrix multiplication

For the same matrices,

```python
A * B
# [[ 5, 12],
#  [21, 32]]

A @ B
# [[19, 22],
#  [43, 50]]
```

`*` pairs entries in the same position. `@` makes row-column dot products. Confusing them changes both the mathematics and analytical meaning.

## 8. Multiple weighted outcomes

Put two weight vectors into columns of $W$:

\[
W=\begin{bmatrix}2&1.5\\0.1&0.25\\0.5&0.1\end{bmatrix}.
\]

One invented rule emphasizes engagement; another emphasizes customer value. Then $XW$ transforms every customer's three features into two scores. With six customers, $(6\times3)(3\times2)=(6\times2)$.

This is the intuition behind a **linear transformation**: multiplication transforms coordinates into new coordinates using weighted combinations. In business terms, a matrix can transform many observations into many outputs in one operation.

These scores are demonstrations, not statistically valid models. Mathematical weighting is not automatically valid business modeling. Weights might eventually come from business judgment, estimation, optimization, statistical modeling, or machine learning; the weights here were chosen only to reveal mechanics.

## 9. Identity and order

The identity matrix behaves like scalar 1:

\[
I=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad AI=A.
\]

Matrix multiplication is generally not commutative. For

\[
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
D=\begin{bmatrix}1&0\\1&1\end{bmatrix},
\]

\[
AD=\begin{bmatrix}3&2\\7&4\end{bmatrix},\qquad
DA=\begin{bmatrix}1&2\\4&6\end{bmatrix}.
\]

Thus $AD\ne DA$. Order matters because the sequences of weighted combinations—or transformations—differ.

## 10. pandas and NumPy implementation

```python
import numpy as np
from analytics_foundations.datasets import load_chapter_07_data

df = load_chapter_07_data()
feature_columns = [
    "visits",
    "average_order_value",
    "months_as_customer",
]
X = df[feature_columns].to_numpy(dtype=float)
weights = np.array([2.0, 0.1, 0.5])

X.shape              # observations, features
X.ndim               # 2
X.T                   # transpose
X[0, :]               # one observation
X[:, 0]               # one feature
X @ weights           # every customer score
np.matmul(X, weights) # equivalent
```

## 11. Previewing regression, not teaching it

Many later models use

\[
\hat{\mathbf y}=X\boldsymbol\beta,
\]

where $X$ contains observed features, $\boldsymbol\beta$ contains coefficients/weights, and $\hat{\mathbf y}$ contains predicted outcomes. This chapter does not teach regression. The useful recognition is simply: **that expression is matrix-vector multiplication**.

## Mastery checkpoints

### Concept checkpoint

1. What do rows and columns of a data matrix represent?
2. What does $x_{ij}$ mean, and why do NumPy indices differ by one?
3. What does matrix shape represent? What happens during transpose?
4. Why must inner dimensions match for multiplication?
5. How is matrix-vector multiplication related to dot products?
6. Why is `A * B` different from `A @ B`?

### Hand-calculation checkpoint

Given

\[
X=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix},\qquad
\mathbf w=\begin{bmatrix}2\\3\end{bmatrix},
\]

state the shapes of $X$ and $\mathbf w$; calculate $X\mathbf w$; calculate $X^T$; and determine the shape of $X^TX$ without calculating it. Checks: $X$ is $3\times2$, $\mathbf w$ has two entries (or $2\times1$ as a column), $X\mathbf w=[8,18,28]^T$, $X^T$ is $2\times3$, and $X^TX$ is $2\times2$. Why $X^TX$ matters is deferred.

### Execution checkpoint

1. Select a different DataFrame column set, create a matrix, and inspect its shape.
2. Define another compatible weight vector and calculate every score without loops.
3. Reproduce one result manually.
4. Combine two weight vectors as columns of a weight matrix and compute all outputs with one `@`.

### Interpretation checkpoint

> A score ranks Customer A above Customer B. Does that prove A is more valuable?

No. The ranking depends on selected features and invented weights; validity requires evidence and business justification.

> Why is matrix multiplication useful before advanced statistics?

It carries out repeated weighted calculations across structured observations compactly and consistently.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** covariance, regression, and multivariable methods can be written compactly with matrices.
- **BUAD 512B — Business Modeling with Python:** NumPy arrays represent numerical datasets and perform efficient linear algebra.
- **BUAD 5112 — Competing Through Business Analytics:** combining measurements into metrics or models requires justification of variables and weights, not merely a valid calculation.

This independent preparation chapter does not reproduce any William & Mary course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-07
```

It loads six fictional customers, converts labeled features to a matrix, inspects rows/columns/entries, transposes a small matrix, checks dimensions, calculates one and multiple scores, contrasts `*` with `@`, handles incompatibility without crashing, and writes table-to-matrix and weighted-score figures.
