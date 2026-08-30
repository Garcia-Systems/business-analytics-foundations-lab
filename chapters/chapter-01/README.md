# Chapter 1 — Functions Become Models

> **Central question:** How can we represent a business relationship mathematically so that we can calculate with it, visualize it, and reason about it?

Chapter 0 turned a business question into an investigation. This chapter supplies a mathematical language for describing the relationships such investigations uncover. A function is not merely an algebra exercise: it is a model that accepts an input and produces an output. Equations, tables, graphs, and Python can all express the **same relationship**.

## Learning objectives

By the end, you should be able to explain inputs and outputs; identify independent and dependent variables; read and evaluate function notation; distinguish a mathematical function from Python code; build a value table and graph; interpret a linear slope and intercept; describe a business domain and range; combine revenue and cost into profit; calculate mathematical and operational break-even; evaluate arrays with NumPy; and explain why a model is useful but incomplete.

## 1. The business problem

A fictional pop-up bakery sells treat boxes. Let

- \(q\) = boxes sold,
- \(p=12\) = price per box,
- \(F=300\) = fixed setup cost, and
- \(v=5\) = variable cost per box.

Management wants to know how revenue, cost, and profit depend on sales volume—and how many boxes must sell before the event becomes profitable. We deliberately ignore many real complications so the relationship remains visible.

## 2. From quantities to relationships

A **variable** names a quantity that can vary. The symbol \(q\) stands for quantity sold; the statement \(q=50\) assigns it one specific value. Price, fixed cost, and variable cost are held constant in this experiment, while quantity changes.

A single calculation answers one case. A relationship answers every case covered by the model. Instead of calculating only revenue for 50 boxes, we write

\[
R(q)=12q.
\]

This says revenue depends on quantity. Quantity is the **independent variable** (the input we choose); revenue is the **dependent variable** (the output determined by that input).

## 3. What is a function?

A function is a rule that associates each allowed input with exactly one output. In

\[
y=f(x),
\]

- \(x\) is the input,
- \(f\) names the function,
- \(f(x)\) means “the output of \(f\) at input \(x\),”
- \(y\) names that output, and
- \(=\) says the two expressions have the same value.

The business notation \(R(q)\) means **revenue as a function of quantity**. \(R\) is the entire rule; \(R(50)\) is the rule evaluated at 50 boxes. They are not interchangeable.

The mapping

\[
q \longrightarrow R(q)
\]

connects inputs to outputs:

| Quantity \(q\) | Revenue \(R(q)\) |
| ---: | ---: |
| 0 | $0 |
| 10 | $120 |
| 20 | $240 |
| 30 | $360 |

## 4. Revenue as a function

In general, fixed-price revenue is \(R(q)=pq\). Here,

\[
R(q)=12q.
\]

Mathematics and code represent the same rule:

```python
def revenue(q):
    return 12 * q
```

The equation is a mathematical object; the Python function is an executable implementation of it. Both map quantity to revenue. On a graph, quantity belongs on the horizontal axis and revenue on the vertical axis. The business meaning is immediate: each additional box adds $12 of revenue **under this model**.

## 5. Cost as a function

Total cost combines setup cost and per-box cost:

\[
C(q)=F+vq=300+5q.
\]

This has the linear form \(y=mx+b\). In this context:

- the intercept \(b=300\) is the cost at zero sales—the fixed cost;
- the slope \(m=5\) means output increases $5 when input increases by one box.

We need no abstract treatment of slope yet. Here it simply answers, “How much does the output change for one more input?” Python expresses the same rule:

```python
def cost(q):
    return 300 + 5 * q
```

## 6. Profit combines models

Models can be built from other models. Profit is revenue less cost:

\[
\begin{aligned}
P(q) &= R(q)-C(q)\\
     &= 12q-(300+5q)\\
     &= 7q-300.
\end{aligned}
\]

The $7 slope is the **contribution per box** toward recovering fixed cost and then earning profit. The corresponding Python preserves the relationship between models:

```python
def profit(q):
    return revenue(q) - cost(q)
```

## 7. Hand-worked calculations

Before executing code, substitute values by hand. At \(q=40\):

\[
R(40)=12(40)=480,
\]

\[
C(40)=300+5(40)=300+200=500,
\]

\[
P(40)=R(40)-C(40)=480-500=-20.
\]

The model predicts a $20 loss. Now predict the sign of profit at 60 boxes before continuing:

\[
R(60)=12(60)=720,
\]

\[
C(60)=300+5(60)=300+300=600,
\]

\[
P(60)=720-600=120.
\]

The prediction is positive: the model gives a $120 profit.

## 8. Domain and range: possible versus meaningful

Algebraically, \(12q\) accepts negative numbers, decimals, and arbitrarily large numbers. But this bakery cannot sell −5 boxes. If boxes are indivisible, the meaningful domain is

\[
q\in\{0,1,2,\ldots\}.
\]

Capacity might narrow it further. The **domain** is the set of meaningful inputs. The **range** is the corresponding set of outputs. Over whole quantities from 0 through 100, revenue's range is $0, $12, …, $1,200; it is not every dollar in between.

This distinction is essential: an input may be **mathematically possible** for an expression but not **meaningful in the model**. A graph's continuous line helps us reason and locate intersections, even though operations occur only at whole-number points.

## 9. From equation to table to graph

Evaluating all three rules produces a table:

| \(q\) | \(R(q)\) | \(C(q)\) | \(P(q)\) |
| ---: | ---: | ---: | ---: |
| 0 | $0 | $300 | −$300 |
| 20 | $240 | $400 | −$160 |
| 40 | $480 | $500 | −$20 |
| 43 | $516 | $515 | $1 |
| 60 | $720 | $600 | $120 |

Each row evaluates equations. Each plotted point locates the same input-output pair. The revenue line starts at zero and rises $12 per box. The cost line starts at $300 and rises $5 per box. Their intersection is break-even. The representations change; the model does not.

## 10. Break-even

Break-even asks where revenue equals cost:

\[
R(q)=C(q)
\]

\[
12q=300+5q
\]

Subtract \(5q\) from both sides:

\[
7q=300
\]

Divide by 7:

\[
q=\frac{300}{7}\approx42.86.
\]

At that mathematical quantity, both revenue and cost are approximately $514.29 and profit is zero. But the bakery cannot sell 0.86 of a box. At 42 boxes profit is −$6; at 43 it is $1. Therefore:

- **mathematical break-even:** approximately 42.86 boxes;
- **operational break-even:** 43 whole boxes, the first feasible quantity at which the model is no longer loss-making (and, here, is profitable).

Algebra answers the mathematical question; business context tells us how to act on the answer.

## 11. Make the model executable

The implementation in `analytics_foundations.chapter_01` deliberately mirrors the notation:

```python
def revenue(quantity):
    return 12 * quantity


def cost(quantity):
    return 300 + 5 * quantity


def profit(quantity):
    return revenue(quantity) - cost(quantity)
```

Calling `revenue(50)` is the code counterpart of evaluating \(R(50)\). A Python function is an implementation and can contain bugs; the mathematical function is the intended relationship. Testing known hand calculations helps establish that the implementation matches it.

## 12. Evaluate many inputs with NumPy

NumPy can pass an array through these unchanged rules:

```python
import numpy as np

quantities = np.arange(0, 101)
revenues = revenue(quantities)
costs = cost(quantities)
profits = profit(quantities)
```

`np.arange(0, 101)` creates whole numbers 0 through 100. Multiplication and addition operate element by element, so one function call evaluates 101 observations. This is a gentle preview of **vectorized computation**, treated formally in Chapter 9. No loop or special model class is needed.

## 13. See and interpret the model

Run the experiment to generate two separate figures:

```bash
python3 -m analytics_foundations chapter-01
```

`figures/chapter-01-revenue-and-cost.png` shows the $300 cost intercept, both slopes, and the annotated intersection. `figures/chapter-01-profit.png` makes negative and positive profit visible around the zero line. The model says the event crosses from a loss at 42 boxes to a profit at 43 boxes.

It does **not** say 43 boxes will actually sell. Nor does it say price, costs, or capacity will remain constant.

## 14. Assumptions and limitations

This model assumes:

- price stays $12 regardless of quantity;
- variable cost stays $5 per box;
- fixed cost stays $300;
- every box is identical, made, and sold;
- capacity constraints, taxes, waste, discounts, uncertainty, and demand are absent.

Models are useful because they simplify reality, not because they reproduce it perfectly. A useful model makes assumptions visible and remains appropriate to the decision. The simplicity here isolates the relationship between quantity and profit; it also limits what management may conclude.

## 15. Experiment

Run the complete experiment from the repository root:

```bash
python3 -m analytics_foundations chapter-01
```

Read the equations, compare the selected quantities, verify the two break-even answers, and inspect both figures. Then trace one value—say 40 boxes—from equation to table row, Python output, graph point, and business interpretation. These are five views of one relationship.

## 16. Mastery checkpoints

### Concept checkpoint

1. What is the input to \(R(q)\), and which quantity is dependent?
2. What is the difference between \(R\) and \(R(50)\)?
3. Why can a negative quantity be accepted algebraically but rejected by this model's business domain?
4. What does the intercept of \(C(q)=300+5q\) represent?
5. What does its slope represent? Include units in your answer.

### Hand-calculation checkpoint

A caterer uses \(R(q)=15q\) and \(C(q)=400+6q\). Without Python:

1. calculate revenue at 30 units;
2. calculate cost at 30 units;
3. calculate profit at 30 units;
4. solve \(R(q)=C(q)\), then report both mathematical and whole-unit operational break-even.

Show substitutions and interpret negative or positive profit in dollars.

### Execution checkpoint

Change one assumption at a time in the experiment: increase price, increase variable cost, then increase fixed cost. Regenerate both plots after each change. Predict the direction of break-even movement first, record the result, and explain why it moved. Do not change several inputs together until you understand each effect.

### Interpretation checkpoint

> Management raises price from $12 to $15. The model predicts a lower break-even quantity. Can we conclude that raising price will improve actual profitability?

No. The calculation is conditional on unchanged sales quantity and other assumptions. This model contains no relationship between price and demand; customers might buy fewer boxes at $15. The result motivates a demand question rather than proving the decision is profitable.

## 17. Why this matters for the W&M preparation path

### BUAD 512B — Business Modeling with Python

Functions turn assumptions into executable models, while NumPy lets the same models operate over arrays of inputs.

### BUAD 512A — Probability & Statistics with R

Probability distributions, statistical estimators, and regression models will later be expressed using mathematical functions.

### BUAD 5112 — Competing Through Business Analytics

Analytics depends on translating business relationships into measurable models while communicating assumptions and limitations. This independent preparation chapter does not reproduce or teach any W&M course.

## 18. Where we go next

Later chapters will add richer mathematical tools. For now, retain the central lesson:

> A function represents a relationship. An equation, value table, graph, Python function, and business interpretation can all express that same model.

Chapter 1 stops here; Chapter 2 remains planned.
