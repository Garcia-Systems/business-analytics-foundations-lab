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
2. Exponents, Logs & Growth
3. Summation & Aggregation
4. Change & Derivatives
5. Accumulation & Integrals
6. Vectors: Data Becomes Geometry
7. Matrices: Data Becomes Structure
8. Linear Algebra for Models

### Part II — Python Becomes an Analytics Tool

9. Arrays & Vectorized Thinking
10. Tables & DataFrames
11. Messy Data
12. Transform, Group & Join
13. Seeing Data

### Part III — Probability as a Model of Uncertainty

14. Events & Probability
15. Conditional Probability
16. Random Variables
17. Distributions
18. Expected Value & Variability
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

Chapters 2–35 and the capstone are **planned**. The planned capstone will integrate business framing, data preparation, quantitative reasoning, computation, visualization, and communication in one coherent analytics challenge.

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

The Chapter 0 experiment loads version-controlled data and writes
`figures/chapter-00-revenue-by-date.png`. Chapter 1 writes separate revenue/cost and profit
figures. Future planned experiments will use the same `chapter-NN` command convention.
