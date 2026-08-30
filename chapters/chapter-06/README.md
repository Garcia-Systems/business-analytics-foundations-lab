# Chapter 6 — Vectors: Data Becomes Geometry

> **Central outcome:** A business observation with several numerical features can be represented as a vector. Once data becomes vectors, geometry gives us mathematical ways to describe magnitude, difference, distance, direction, and weighted combinations.

Earlier chapters treated individual quantities. This chapter asks: **How can several measurements about one observation be represented and compared mathematically?** Ten fictional Harbor Cafe customers travel from business entities → features → vectors → points → operations → careful interpretation. We do not use clustering.

## Learning objectives

By the end, you should be able to explain vectors, components, dimension, and direction; distinguish scalars and vectors; represent a table row as a feature vector and 2D point; add, subtract, and scale vectors; calculate magnitude, Euclidean distance, and dot products; interpret distance and weighted combinations; recognize feature-scale problems; and use pandas and NumPy. These are building blocks for matrices and regression, not a lesson on either topic.

## 1. Scalars, vectors, and components

A **scalar is one numerical value**, such as $x=42$. A **vector contains an ordered collection of numerical components**:

\[
\mathbf{x}=\begin{bmatrix}5\\30\end{bmatrix},\qquad x_1=5,\quad x_2=30.
\]

Boldface helps distinguish the vector $\mathbf{x}$ from a scalar $x$, although typography varies across textbooks and software. The subscript identifies a component. In Chapter 3, $x_i$ often identified observation $i$; here it can identify component $i$ of one vector. Notation gets meaning from context.

Customer A has 5 visits and a $30 average order value, so

\[
\mathbf{x}_A=\begin{bmatrix}5\\30\end{bmatrix}.
\]

Customer B has 7 visits and a $25 average order value, so $\mathbf{x}_B=[7,25]^T$; the transpose symbol compactly displays a column vector inline.

| customer_id | visits | average_order_value |
|---|---:|---:|
| A | 5 | 30 |
| B | 7 | 25 |

The row `(A, 5, 30)`, vector $[5,30]^T$, and point $(5,30)$ are **different representations of the same data**. This parallels equation/table/graph/code equivalence.

## 2. Dimension and geometry

Two real components mean $\mathbf{x}_A\in\mathbb{R}^2$: $\mathbb R$ means real numbers and superscript 2 means two components, not a square. Adding months as a customer gives $[5,30,18]^T\in\mathbb{R}^3$. Generally, an observation with $p$ selected features is $\mathbf{x}\in\mathbb{R}^p$. This is feature-count notation, not formal vector-space theory.

Put visits on the horizontal axis and average order value on the vertical axis. $[5,30]^T$ can be a **point** at $(5,30)$ or an **arrow** from $(0,0)$ to $(5,30)$. A point emphasizes location; an arrow emphasizes magnitude and direction. The point interpretation is often especially useful for analytical rows.

## 3. Addition, subtraction, and scaling

For $\mathbf a=[2,3]^T$ and $\mathbf b=[4,1]^T$, addition works component-wise:

\[
\mathbf a+\mathbf b=\begin{bmatrix}2+4\\3+1\end{bmatrix}=\begin{bmatrix}6\\4\end{bmatrix}.
\]

Geometrically, placing $\mathbf b$ at the head of $\mathbf a$ reaches the sum. We need not force a business story onto every elementary operation.

Subtraction creates displacement:

\[
\mathbf{x}_B-\mathbf{x}_A=\begin{bmatrix}7-5\\25-30\end{bmatrix}=\begin{bmatrix}2\\-5\end{bmatrix}.
\]

B visits twice more but spends $5 less per order. The arrow from A to B is this difference.

A scalar multiplies every component:

\[
2\begin{bmatrix}3\\4\end{bmatrix}=\begin{bmatrix}6\\8\end{bmatrix}.
\]

A positive scalar preserves direction and changes length; a negative scalar also reverses direction; zero collapses the arrow to the origin.

## 4. Magnitude and distance

Euclidean magnitude is vector length. By the Pythagorean theorem,

\[
\|\mathbf{x}\|=\sqrt{x_1^2+x_2^2}.
\]

Calculate by hand before using a library:

\[
\left\|\begin{bmatrix}3\\4\end{bmatrix}\right\|=\sqrt{3^2+4^2}=5.
\]

Distance makes magnitude analytically useful:

\[
d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}.
\]

For A and B, displacement is $[2,-5]^T$, so $d(A,B)=\sqrt{2^2+(-5)^2}=\sqrt{29}\approx5.385$. This is the straight segment between the points. Smaller distance can mean more similar **under the chosen representation and metric**. Similarity depends on selected features and scale.

## 5. Feature scale changes geometry

Suppose an anchor customer is $[2,100]^T$: 2 visits and $100 annual spending. Compare frequent $[9,110]^T$ and spend-alike $[3,500]^T$:

\[
d(\text{anchor},\text{frequent})=\sqrt{7^2+10^2}\approx12.21,
\]
\[
d(\text{anchor},\text{spend-alike})=\sqrt{1^2+400^2}\approx400.00.
\]

Raw dollars dominate. Express spending in thousands and the points become $[2,.1]^T$, $[9,.11]^T$, and $[3,.5]^T$. Distances become about 7.00 and 1.08: the nearest comparison reverses. Customers did not change; representation did.

> Distances are affected by measurement scale. Geometry depends on how the data is represented.

Standardization is deliberately deferred. For now, notice the problem rather than silently treating raw units as comparable.

## 6. Dot products: multiply, then sum

\[
\mathbf{x}\cdot\mathbf{y}=x_1y_1+x_2y_2+\cdots+x_ny_n=\sum_{i=1}^{n}x_i y_i.
\]

This reconnects to Chapter 3: **multiply corresponding components, then sum**. For $[2,3]^T$ and $[4,1]^T$, the dot product is $2(4)+3(1)=11$.

A practical dot product is a weighted combination. With quantities $\mathbf q=[10,4,2]^T$ and prices $\mathbf p=[5,8,12]^T$,

\[
\mathbf q\cdot\mathbf p=10(5)+4(8)+2(12)=106.
\]

Each quantity receives its price as a weight. Later, $\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots$ will use this same weighted-combination structure. That motivates regression without teaching it.

Direction offers another interpretation:

\[
\mathbf{x}\cdot\mathbf{y}=\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta.
\]

A positive result means broadly similar direction, zero means perpendicular, and negative means broadly opposite. No trigonometric derivation is needed. **Orthogonal** means perpendicular: $[1,0]^T\cdot[0,1]^T=0$. This will matter again in linear algebra and regression.

## 7. pandas row → NumPy array → mathematical vector

```python
df = load_chapter_06_data()
features = df[["visits", "average_order_value"]]
customer = features.iloc[0].to_numpy(dtype=float)
# array([ 5., 30.])
```

Feature ordering defines component meaning. A third selected column produces a 3D vector even when a page cannot display it directly.

```python
import numpy as np
x = np.array([5.0, 30.0])
y = np.array([7.0, 25.0])
x + y                 # element-wise addition
y - x                 # displacement
2 * x                 # scalar multiplication
np.linalg.norm(x)     # magnitude, after the manual formula
np.linalg.norm(y - x) # distance
np.dot(x, y)          # dot product
x @ y                 # same dot product for these 1D arrays
x * y                 # element-wise products: NOT the dot product
```

For 1D arrays, `*` returns component-wise products while `@` performs the dot-product/matrix-multiplication operation and returns a scalar here. This distinction becomes extremely important in Chapter 7.

## 8. Business interpretation

Customers close in visits and order value may differ in tenure, product preferences, profitability, geography, or promotion response.

> Distance is not an intrinsic property of customers. It is a property of the representation and metric we chose.

Euclidean distance assumes one geometry. Feature selection encodes relevance and scale encodes relative influence. Clean data does not remove those judgment calls.

## Mastery checkpoints

### Concept checkpoint

1. What distinguishes a scalar and vector? What does dimension mean?
2. How can a DataFrame row become a vector?
3. What do subtraction and magnitude represent?
4. How is distance calculated from subtraction and magnitude?
5. Why can scale distort distance?
6. What does a dot product do computationally?

### Hand-calculation checkpoint

Let $\mathbf a=[3,4]^T$ and $\mathbf b=[6,8]^T$. Calculate $\mathbf a+\mathbf b$, $\mathbf b-\mathbf a$, $2\mathbf a$, $\|\mathbf a\|$, $\|\mathbf b-\mathbf a\|$, and $\mathbf a\cdot\mathbf b$. Checks: $[9,12]^T$, $[3,4]^T$, $[6,8]^T$, 5, 5, and 50.

### Execution checkpoint

1. Select different customer features and convert rows to vectors.
2. Calculate pairwise distances manually and with NumPy; identify a nearest customer.
3. Rescale one feature and observe whether the answer changes.
4. Reproduce a dot product with element-wise multiplication and `sum`, without `np.dot`.

### Interpretation checkpoint

> If A is closest to B using visits and order value, are they most similar overall?

No. The conclusion depends on features, scale, and metric; important omitted differences may remain.

> Why is a dot product useful when a model assigns different importance to features?

It multiplies each feature by its own weight and sums the contributions.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** datasets can be collections of vectors; covariance and regression eventually rely on vector and matrix representations.
- **BUAD 512B — Business Modeling with Python:** NumPy arrays computationally represent vectors and efficiently support numerical operations.
- **BUAD 5112 — Competing Through Business Analytics:** feature choice determines which comparisons and conclusions a representation supports.

This independent preparation chapter does not reproduce any William & Mary course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-06
```

It previews data, constructs vectors, calculates displacement, magnitude, distance, the closest raw pair, a scale reversal, a dot product, and weighted revenue. It generates customer-point, distance, and feature-scale figures and states assumptions.
