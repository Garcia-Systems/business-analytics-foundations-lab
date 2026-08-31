# Business Analytics Foundations Lab

Business Analytics Foundations Lab is an executable preparation textbook for students approaching graduate study in business analytics. It develops the mathematical, statistical, programming, data, visualization, and business-reasoning prerequisites that make advanced analytics coursework productive rather than overwhelming.

The project is designed with William & Mary's business analytics curriculum in mind—especially the foundations underlying BUAD 512A, BUAD 512B, and BUAD 5112—but it does **not** reproduce those courses or represent university course material. It is an independent preparation path for prospective or incoming graduate students, career changers, and anyone who wants a rigorous bridge into business analytics.

## How the executable textbook works

Every completed chapter will follow the same learning loop:

> business question → intuition → mathematics → small hand-worked example → executable implementation → visualization → business interpretation → mastery checkpoint

A chapter may contain explanatory Markdown with mathematical notation, executable Python or a focused notebook, generated figures, small version-controlled datasets, and exercises or checkpoints. Each chapter can also register one small experiment with the command-line interface. The registry is intentionally explicit and lightweight: the source remains understandable to a learner and can grow one chapter at a time.

Python 3.12+ is the primary computational environment. R will be introduced deliberately in Part V for statistical work; it is not bundled into the Python environment. This sequencing lets readers first develop transferable analytical ideas, then see how statistical workflows are expressed idiomatically in R.

## Curriculum roadmap

### Part I — Mathematical Language of Analytics

0. **The Analytics Laboratory — implemented**
1. **Functions Become Models — implemented**
2. **Exponents, Logs & Growth — implemented**
3. **Summation & Aggregation — implemented**
4. **Change & Derivatives — implemented**
5. **Accumulation & Integrals — implemented**
6. **Vectors: Data Becomes Geometry — implemented**
7. **Matrices: Data Becomes Structure — implemented**
8. **Linear Algebra for Models — implemented**

### Part II — Python Becomes an Analytics Tool — complete

9. **Arrays & Vectorized Thinking — implemented**
10. **Tables & DataFrames — implemented**
11. **Messy Data — implemented**
12. **Transform, Group & Join — implemented**
13. **Seeing Data — implemented**

### Part III — Probability as a Model of Uncertainty — begun

14. **Events & Probability — implemented**
15. **Conditional Probability — implemented**
16. **Random Variables — implemented**
17. **Distributions — implemented**
18. **Expected Value & Variability — implemented**
19. Covariance & Dependence
20. Monte Carlo Business

### Part IV — Statistics: Learning From Samples

21. Samples Tell Stories
22. Sampling Distributions
23. Estimation & Confidence
24. Hypothesis Testing
25. Comparing Groups
26. Regression as a Model
27. Multiple Regression

### Part V — R for Statistical Work

28. R for Programmers
29. Statistics in R

### Part VI — Thinking Like a Business Analyst

30. From Business Problem to Analytical Question
31. Metrics, KPIs & Decisions
32. Descriptive, Predictive & Prescriptive Analytics
33. Evidence, Uncertainty & Communication
34. Visualization as Argument
35. Analytics, Algorithms & Ethics

### Capstone — James River Analytics Challenge

Chapters 19–35 and the capstone are **planned**. Part III — Probability as a Model of Uncertainty has begun. The planned capstone will integrate business framing, data preparation, quantitative reasoning, computation, visualization, and communication in one coherent analytics challenge.

## Repository layout

```text
chapters/                    Chapter Markdown, code, notebooks, and checkpoints
data/raw/                    Small source datasets worth versioning
data/processed/              Reproducible, chapter-ready datasets
figures/                     Generated output (ignored except for its placeholder)
scripts/                     Small project-level utility scripts
src/analytics_foundations/   CLI registry and reusable teaching code
tests/                       Infrastructure and, later, chapter tests
```

Reusable dataset, visualization, simulation, and statistics code has a named module under `src/analytics_foundations/`. These modules begin deliberately minimal and should gain code only when a chapter establishes a real need.

## Setup and verification

From the repository root, use Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python3 -m analytics_foundations --help
```

After installation, the equivalent console command is also available:

```bash
analytics-foundations --help
```

Chapter 0 is implemented. Run its complete cafe analytics experiment with:

```bash
python3 -m analytics_foundations chapter-00
```

Chapter 1 is also implemented. Run its revenue, cost, profit, and break-even model with:

```bash
python3 -m analytics_foundations chapter-01
```

Chapter 2 compares additive and compound customer growth and uses logarithms to work backward:

```bash
python3 -m analytics_foundations chapter-02
```

Chapter 3 translates summation notation into transaction, NumPy, and pandas aggregations:

```bash
python3 -m analytics_foundations chapter-03
```

Chapter 4 develops derivatives from shrinking average rates of change and interprets marginal business effects:

```bash
python3 -m analytics_foundations chapter-04
```

Chapter 5 builds accumulated totals from rates with Riemann sums, numerical integration, and the Fundamental Theorem:

```bash
python3 -m analytics_foundations chapter-05
```

Chapter 6 turns customer feature rows into vectors and uses geometry for comparison and weighted combinations:

```bash
python3 -m analytics_foundations chapter-06
```

Chapter 7 stacks customer vectors into matrices and applies weighted rules across every row:

```bash
python3 -m analytics_foundations chapter-07
```

Chapter 8 uses systems, rank, and least squares to connect matrices to model coefficients:

```bash
python3 -m analytics_foundations chapter-08
```

Chapter 9 turns restaurant matrices into vectorized metrics, masks, axis aggregations, and broadcast target comparisons:

```bash
python3 -m analytics_foundations chapter-09
```

Chapter 10 uses labeled restaurant transactions to select, filter, derive, group, and preserve business meaning:

```bash
python3 -m analytics_foundations chapter-10
```

Chapter 11 audits deliberately messy restaurant transactions, applies documented cleaning decisions, and validates a reproducible analytical dataset:

```bash
python3 -m analytics_foundations chapter-11
```

Chapter 12 transforms transaction, daily labor, and location tables into a grain-safe analytical dataset:

```bash
python3 -m analytics_foundations chapter-12
```

Chapter 13 uses reproducible Matplotlib views to investigate distributions, comparisons, relationships, time patterns, and visual choices:

```bash
python3 -m analytics_foundations chapter-13
```

Chapter 14 defines events in a restaurant demand probability model and makes uncertainty executable with reproducible simulation:

```bash
python3 -m analytics_foundations chapter-14
```

Chapter 15 uses restaurant contingency tables and fraud alerts to develop conditional probability, independence, Bayes' rule, and conditional simulation:

```bash
python3 -m analytics_foundations chapter-15
```

Chapter 16 maps uncertain restaurant demand to a numerical random variable, then explores its PMF, CDF, transformations, and reproducible simulation:

```bash
python3 -m analytics_foundations chapter-16
```

Chapter 17 introduces Bernoulli, Binomial, Uniform, and Normal models, their parameters, assumptions, probability calculations, and simulation:

```bash
python3 -m analytics_foundations chapter-17
```

Chapter 18 develops probability-weighted expectation, variance, standard deviation, transformations, simulation, and business risk comparisons:

```bash
python3 -m analytics_foundations chapter-18
```

The Chapter 0 experiment loads version-controlled data and writes
`figures/chapter-00-revenue-by-date.png`. Chapter 1 writes separate revenue/cost and profit
figures. Chapter 2 writes linear-versus-exponential and repeated-multiplication figures. Chapter 3 writes revenue-by-category and category-contribution figures. Chapter 4 writes average-rate, secant-to-tangent, and profit-versus-marginal-profit figures. Chapter 5 writes accumulation, Riemann-rectangle, refinement, trapezoid, Simpson, and signed-accumulation figures. Chapter 6 writes customer-point, distance, and feature-scale figures. Chapter 7 writes table-to-matrix and weighted-score figures. Chapter 8 writes unique, dependent, inconsistent-system, and least-squares figures. Chapter 9 writes revenue-matrix, location-total, and target-deviation figures. Chapter 10 writes category, location, and daily-revenue figures. Chapter 11 writes missingness, category-standardization, outlier-candidate, and raw-versus-cleaned reconciliation figures. Chapter 12 writes grain-transformation, revenue-and-labor, and revenue-per-labor-hour figures. Chapter 13 writes distribution, group-comparison, relationship, time, grain-comparison, promotion, and axis-skepticism figures. Chapter 14 writes demand-probability, event, complement, union, intersection, and simulation-convergence figures. Chapter 15 writes restricted-denominator, contingency-table, probability-tree, and Bayes-count figures. Chapter 16 writes outcome-mapping, PMF, CDF, simulation-versus-model, and continuous-preview figures. Chapter 17 writes Bernoulli and Binomial PMFs, Uniform and Normal density areas, Normal parameter comparisons, and theory-versus-simulation figures. Chapter 18 writes weighted-balance, same-mean risk, deviation, cumulative-mean, and empirical-variance figures. Future planned experiments will use the same `chapter-NN` command convention.
