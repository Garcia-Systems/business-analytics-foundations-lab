# Chapter 5 — Accumulation & Integrals

> **Central outcome:** An integral accumulates many small contributions. Riemann sums make that idea visible, numerical integration makes it executable, and the Fundamental Theorem connects accumulation back to the rates of change studied in Chapter 4.

Chapter 4 asked, “How quickly is something changing?” This chapter asks the complementary question: **If we know the rate, how much accumulates over an interval?** Derivatives break change down into local rates. Integrals build local contributions back into accumulated totals. We begin with an accumulation problem—not antiderivative rules.

## Learning objectives

By the end, you should be able to distinguish a quantity from its rate; explain rate × interval; partition an interval and calculate $\Delta t$; calculate left, right, and midpoint sums; interpret a definite integral and its notation; reason about signed accumulation; use trapezoidal and Simpson approximations; use a simple antiderivative and the Fundamental Theorem; implement and compare these methods in NumPy and SciPy; track units; and interpret the result with appropriate business caution.

## 1. Begin with units

Suppose customers arrive at a constant rate of 10 customers/hour for 3 hours. Then

\[
10\frac{\text{customers}}{\text{hour}}\times3\text{ hours}=30\text{ customers}.
\]

The time units cancel:

\[
\frac{\text{customers}}{\text{hour}}\times\text{hour}=\text{customers}.
\]

That is the intuition for integration. **What if the arrival rate changes throughout the three hours?**

Fictional **Harbor Cafe** models its continuously varying arrival rate as

\[
r(t)=12+6t-t^2,\qquad 0\le t\le6,
\]

where $t$ is hours after opening and $r(t)$ is measured in customers/hour. This positive, smooth curve begins at 12, peaks at 21 after 3 hours, and returns to 12 after 6 hours. Crucially, $r(t)$ is **not a number of customers**. Our question is:

> If customer arrivals vary continuously, approximately how many customers arrive between opening and six hours later?

Pretend initially that we know the rate only at selected times.

## 2. Partition time into small intervals

Divide $[a,b]$ into $n$ equal subintervals. Their common width is

\[
\Delta t=\frac{b-a}{n}.
\]

Here $a$ is the starting time, $b$ the ending time, $n$ the number of pieces, and $\Delta t$ (delta t) the duration of each piece. For $[0,6]$ and $n=6$,

\[
\Delta t=\frac{6-0}{6}=1\text{ hour}.
\]

On one short interval,

\[
\text{customers}\approx\text{arrival rate}\times\text{time width}=r(t_i)\Delta t.
\]

Geometrically, width is hours, height is customers/hour, and rectangle area is customers. Thus “area under the curve” is not an arbitrary geometry trick: **rate × interval = quantity**.

## 3. Left, right, and midpoint rectangles

A left sum samples the left end of each interval:

\[
L_n=\sum_{i=0}^{n-1}r(t_i)\Delta t.
\]

For $n=3$, $\Delta t=2$ and the left times are 0, 2, 4. Since $r(0)=12$, $r(2)=20$, and $r(4)=20$,

\[
L_3=[12+20+20](2)=104\text{ customers}.
\]

A right sum uses times 2, 4, 6:

\[
R_3=[r(2)+r(4)+r(6)](2)=[20+20+12](2)=104.
\]

They happen to agree because this curve is symmetric—not because left and right sums always agree. On an increasing piece, a left rectangle tends to be low and a right rectangle high; on a decreasing piece the roles reverse. Never use the simplistic rule “left is always low.”

A midpoint sum samples centers 1, 3, 5:

\[
M_3=[r(1)+r(3)+r(5)](2)=[17+21+17](2)=110.
\]

Sampling near the center can balance endpoint error. The generated left, right, and midpoint figures use the same $n=6$ partition so the error is visually comparable.

## 4. Increasing $n$: a numerical experiment

| $n$ | $\Delta t$ | Left | Right | Midpoint | $|$midpoint error$|$ |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 104.0000 | 104.0000 | 110.0000 | 2.0000 |
| 6 | 1 | 107.0000 | 107.0000 | 108.5000 | 0.5000 |
| 12 | 0.5 | 107.7500 | 107.7500 | 108.1250 | 0.1250 |
| 24 | 0.25 | 107.9375 | 107.9375 | 108.0312 | 0.0312 |
| 48 | 0.125 | 107.9844 | 107.9844 | 108.0078 | 0.0078 |

The exact value, established later, is 108. As $n\uparrow$, $\Delta t\downarrow$ and the rectangles usually follow a smooth curve more closely. This is convergence, not a guarantee that every method improves monotonically for every possible function.

## 5. The definite integral emerges

We now name the exact continuous accumulation:

\[
\int_a^b r(t)\,dt.
\]

- $\int$ is the elongated-s symbol for continuous summation/accumulation.
- $a$ is the start and $b$ the end.
- $r(t)$ is the rate accumulated.
- $dt$ says that time is partitioned and accumulated.

Conceptually,

\[
\int_a^b r(t)\,dt=\lim_{n\to\infty}\sum r(t_i^*)\Delta t.
\]

$t_i^*$ may be a selected point in each piece. As partitions become arbitrarily fine, the sums approach exact modeled accumulation.

## 6. Trapezoids: sloping rather than flat tops

A horizontal rectangle top may follow a sloping curve poorly. Connecting adjacent endpoint heights gives a trapezoid. On one interval,

\[
\text{area}\approx\frac{f(x_i)+f(x_{i+1})}{2}\Delta x.
\]

For $n=3$, averaging adjacent rates gives

\[
T_3=2\left[\frac{12+20}{2}+\frac{20+20}{2}+\frac{20+12}{2}\right]=104.
\]

For equal widths, $T_n=(L_n+R_n)/2$. The composite formula is

\[
T_n=\Delta x\left[\tfrac12f(x_0)+f(x_1)+\cdots+f(x_{n-1})+\tfrac12f(x_n)\right].
\]

## 7. Simpson's rule: curved pieces

Rectangles have flat tops; trapezoids use straight lines; Simpson's rule follows quadratic/parabolic pieces through groups of three equally spaced points. For positive even $n$,

\[
S_n=\frac{\Delta x}{3}[f(x_0)+4f(x_1)+2f(x_2)+4f(x_3)+\cdots+4f(x_{n-1})+f(x_n)].
\]

The weights are $1,4,2,4,2,\ldots,4,1$: endpoints appear once, odd interior nodes receive weight 4, and even interior nodes weight 2 as neighboring parabolic pieces meet. For $n=6$, $\Delta t=1$, rates are $12,17,20,21,20,17,12$, so

\[
S_6=\frac13[12+4(17)+2(20)+4(21)+2(20)+4(17)+12]=108.
\]

This quadratic happens to be integrated exactly by Simpson's rule. That does not make Simpson universally best; smoothness, spacing, data quality, and function behavior matter.

A precise midpoint/trapezoid relationship is also useful. If $T_n$ uses $n$ panels of width $h$ and $M_n$ uses their $n$ midpoints, then Simpson over the refined $2n$ panels of width $h/2$ is

\[
S_{2n}=\frac{T_n+2M_n}{3}.
\]

This is a weighted combination with compatible nodes and widths—not a simple average.

## 8. Exact integration, after approximation

An **antiderivative** $R$ has derivative $R'=r$. Here

\[
R(t)=12t+3t^2-\frac{t^3}{3},
\]

because $R'(t)=12+6t-t^2$. The Fundamental Theorem of Calculus gives

\[
\int_0^6r(t)\,dt=R(6)-R(0)
=\left(72+108-72\right)-0=108\text{ customers}.
\]

Differentiation asks for the rate from an accumulated quantity. Integration reconstructs accumulated change from the rate. This is the Chapter 4–5 connection.

## 9. Signed accumulation is net accumulation

Negative arrival rates would be unrealistic, so use a separate net cash-flow model (hundreds of dollars/day):

\[
c(t)=t-2,\qquad0\le t\le4.
\]

The rate is below zero for two days (cash leaving) and above zero for two days (cash entering). The negative triangle has signed area $-\tfrac12(2)(2)=-2$ and the positive triangle has area $+2$, so

\[
\int_0^4c(t)\,dt=0.
\]

That is **signed/net area**, not total geometric area. Total magnitude is $2+2=4$ hundred dollars and is represented by $\int_0^4|c(t)|\,dt$. Values below the horizontal axis contribute negatively because they reverse the accumulated quantity.

## 10. Transparent Python, then a library check

```python
def left_sum(f, a, b, n):
    dx = (b - a) / n
    x = a + np.arange(n) * dx
    return np.sum(f(x)) * dx


def simpson_rule(f, a, b, n):
    if n <= 0 or n % 2:
        raise ValueError("Simpson's rule requires a positive even n")
    dx = (b - a) / n
    y = f(np.linspace(a, b, n + 1))
    return dx / 3 * (y[0] + y[-1]
                     + 4 * np.sum(y[1:-1:2])
                     + 2 * np.sum(y[2:-1:2]))
```

The source also implements right, midpoint, and trapezoid methods. Only after those calculations does the experiment compare `scipy.integrate.quad` and the current `scipy.integrate.simpson` API. NumPy supplies transparent arrays of nodes, rates, slices, and weights; Matplotlib makes the geometry visible.

For $n=6$:

| Method | $n$ | Approximation | Absolute error |
|---|---:|---:|---:|
| Left | 6 | 107.0000 | 1.0000 |
| Right | 6 | 107.0000 | 1.0000 |
| Midpoint | 6 | 108.5000 | 0.5000 |
| Trapezoid | 6 | 107.0000 | 1.0000 |
| Simpson | 6 | 108.0000 | 0.0000 |

## 11. Business meaning, units, and limitations

**Why does the integral of customers/hour over hours produce customers?** The time unit cancels. The same structure interprets many business totals:

\[
\frac{\text{dollars}}{\text{day}}\times\text{days}=\text{dollars},\quad
\frac{\text{units}}{\text{hour}}\times\text{hours}=\text{units},\quad
\frac{\text{kWh}}{\text{hour}}\times\text{hours}=\text{kWh}.
\]

The units often tell us what an integral means.

The smooth rate is a model; real arrivals are discrete events and uncertain. **Numerical approximation error** is the difference between a method and the integral of the chosen model. **Model error** is the difference between that mathematical model and reality. Extraordinary Simpson precision cannot make an unrealistic model accurate. Staffing decisions would also need variability, service time, capacity, breaks, and costs.

## Mastery checkpoints

### Concept checkpoint

1. What does $r(t)$ represent? Is it a count?
2. Why does rate × time produce a quantity?
3. What does $\Delta t$ represent? What happens to it as $n$ increases?
4. How do left, right, and midpoint samples differ?
5. What does the definite integral represent, and what does each symbol mean?
6. Why can an integral be negative?
7. How is integration related to differentiation?

### Hand-calculation checkpoint

Suppose measured rates (customers/hour) are:

| Time (hours) | 0 | 0.5 | 1 | 1.5 | 2 |
|---:|---:|---:|---:|---:|---:|
| Rate | 8 | 10 | 13 | 12 | 9 |

Using one-hour panels on $[0,2]$, calculate the left sum from rates at 0 and 1, right sum from 1 and 2, midpoint sum from 0.5 and 1.5, and trapezoidal estimate. State units. (Checks: 21, 22, 22, and 21.5 customers.)

### Execution checkpoint

1. Run the five Riemann values and calculate each method's error.
2. Graph error versus $n$ on an appropriate scale.
3. Change the rate function, predict the effect, and regenerate figures.
4. Compare manual NumPy methods with SciPy; try odd $n$ in Simpson and explain the clean failure.

### Interpretation checkpoint

> A numerical method estimates 318.0001 customers while the exact mathematical integral is 318. Does that mean exactly 318 real customers will arrive?

No. It demonstrates numerical agreement with the chosen continuous model, not certainty about discrete, stochastic arrivals or model validity.

> A cash-flow rate is negative for part of the month. Why doesn't the definite integral simply count that region as positive area?

A definite integral measures signed/net accumulation: outflow reverses cash accumulation. Total magnitude would require integrating the absolute value.

## W&M preparation connection

- **BUAD 512A — Probability & Statistics with R:** continuous probabilities accumulate area under probability-density curves.
- **BUAD 512B — Business Modeling with Python:** numerical methods calculate quantities without convenient closed forms.
- **BUAD 5112 — Competing Through Business Analytics:** business rates accumulate over time, while mathematical precision must remain distinct from model uncertainty.

This independent preparation chapter does not reproduce any William & Mary course.

## Run the experiment

```bash
python3 -m analytics_foundations chapter-05
```

Inspect eight figures: accumulated rate area; separate left, right, and midpoint rectangles; coarse versus fine partitions; trapezoids; Simpson's curved pieces; and signed cash flow.
