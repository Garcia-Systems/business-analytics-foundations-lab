"""Accumulation and numerical integration experiments for Chapter 5."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from analytics_foundations.datasets import PROJECT_ROOT


def arrival_rate(t):
    """Modeled cafe arrivals in customers per hour, ``12 + 6t - t²``."""
    t = np.asarray(t)
    return 12 + 6 * t - t**2


def cash_flow_rate(t):
    """Net cash-flow rate in hundreds of dollars per day."""
    return np.asarray(t) - 2


def interval_width(a: float, b: float, n: int) -> float:
    """Return the width of each of ``n`` equal pieces of [a, b]."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return (b - a) / n


def left_sum(f: Callable, a: float, b: float, n: int) -> float:
    """Approximate accumulation with left-endpoint rectangles."""
    dx = interval_width(a, b, n)
    x = a + np.arange(n) * dx
    return float(np.sum(f(x)) * dx)


def right_sum(f: Callable, a: float, b: float, n: int) -> float:
    """Approximate accumulation with right-endpoint rectangles."""
    dx = interval_width(a, b, n)
    x = a + np.arange(1, n + 1) * dx
    return float(np.sum(f(x)) * dx)


def midpoint_sum(f: Callable, a: float, b: float, n: int) -> float:
    """Approximate accumulation with midpoint rectangles."""
    dx = interval_width(a, b, n)
    x = a + (np.arange(n) + 0.5) * dx
    return float(np.sum(f(x)) * dx)


def trapezoid_rule(f: Callable, a: float, b: float, n: int) -> float:
    """Approximate accumulation with ``n`` equal-width trapezoids."""
    dx = interval_width(a, b, n)
    x = np.linspace(a, b, n + 1)
    return float(dx * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1])))


def simpson_rule(f: Callable, a: float, b: float, n: int) -> float:
    """Apply composite Simpson's rule; ``n`` must be positive and even."""
    interval_width(a, b, n)
    if n % 2:
        raise ValueError("Simpson's rule requires an even number of subintervals")
    dx = (b - a) / n
    y = f(np.linspace(a, b, n + 1))
    return float(dx / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])))


def arrival_antiderivative(t):
    """Return R(t) where R'(t) is the modeled arrival rate."""
    t = np.asarray(t)
    return 12 * t + 3 * t**2 - t**3 / 3


def exact_arrivals(a: float = 0, b: float = 6) -> float:
    """Evaluate the polynomial integral using the Fundamental Theorem."""
    return float(arrival_antiderivative(b) - arrival_antiderivative(a))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def create_accumulation_figure(path: Path) -> Path:
    """Shade the rate-times-time accumulation under the arrival curve."""
    x = np.linspace(0, 6, 500); y = arrival_rate(x)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, y, linewidth=2)
    ax.fill_between(x, 0, y, alpha=.25, label="Accumulation = 108 customers")
    ax.set(title="Arrival rate accumulates customers", xlabel="Hours after opening", ylabel="Arrival rate (customers/hour)")
    ax.grid(alpha=.2); ax.legend(); return _save(fig, path)


def create_riemann_figure(path: Path, method: str, n: int = 6) -> Path:
    """Draw left, right, or midpoint rectangles on the same rate model."""
    choices = {"left": 0, "right": 1, "midpoint": .5}
    if method not in choices: raise ValueError("method must be left, right, or midpoint")
    dx = 6 / n; starts = np.arange(n) * dx
    samples = starts + choices[method] * dx
    x = np.linspace(0, 6, 500)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, arrival_rate(x), color="black", linewidth=2, label="r(t)")
    ax.bar(starts, arrival_rate(samples), width=dx, align="edge", alpha=.35, edgecolor="tab:blue", label=f"{method.title()} rectangles")
    ax.scatter(samples, arrival_rate(samples), color="tab:red", zorder=3, label="sampled rates")
    ax.set(title=f"{method.title()} Riemann sum (n={n})", xlabel="Hours after opening", ylabel="Arrival rate (customers/hour)")
    ax.grid(alpha=.2); ax.legend(); return _save(fig, path)


def create_refinement_figure(path: Path) -> Path:
    """Contrast coarse and fine midpoint partitions."""
    x = np.linspace(0, 6, 500); fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, n in zip(axes, (3, 24), strict=True):
        dx = 6 / n; starts = np.arange(n) * dx; mid = starts + dx / 2
        ax.plot(x, arrival_rate(x), color="black", linewidth=2)
        ax.bar(starts, arrival_rate(mid), width=dx, align="edge", alpha=.35, edgecolor="tab:blue")
        ax.set(title=f"n={n}, Δt={dx:g} hour", xlabel="Hours after opening"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Arrival rate (customers/hour)"); fig.suptitle("More, narrower rectangles fit the curve more closely")
    return _save(fig, path)


def create_trapezoid_figure(path: Path, n: int = 6) -> Path:
    """Show piecewise straight trapezoid tops against the smooth curve."""
    nodes = np.linspace(0, 6, n + 1); x = np.linspace(0, 6, 500)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, arrival_rate(x), color="black", linewidth=2, label="smooth rate")
    ax.fill_between(nodes, 0, arrival_rate(nodes), alpha=.25, label="trapezoids")
    ax.plot(nodes, arrival_rate(nodes), "o-", label="straight-line tops")
    for node in nodes: ax.vlines(node, 0, arrival_rate(node), color="gray", alpha=.4)
    ax.set(title=f"Trapezoidal approximation (n={n})", xlabel="Hours after opening", ylabel="Arrival rate (customers/hour)")
    ax.grid(alpha=.2); ax.legend(); return _save(fig, path)


def create_simpson_figure(path: Path) -> Path:
    """Illustrate Simpson's parabolic pieces using two node triples."""
    x = np.linspace(0, 6, 500); nodes = np.array([0., 1.5, 3., 4.5, 6.])
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, arrival_rate(x), color="black", linewidth=3, label="r(t)")
    for left in (0, 2):
        triple = nodes[left:left + 3]; curve_x = np.linspace(triple[0], triple[-1], 150)
        curve_y = np.polyval(np.polyfit(triple, arrival_rate(triple), 2), curve_x)
        ax.fill_between(curve_x, 0, curve_y, alpha=.18)
    ax.scatter(nodes, arrival_rate(nodes), color="tab:red", zorder=3, label="endpoints and midpoints")
    ax.set(title="Simpson's rule follows parabolic pieces", xlabel="Hours after opening", ylabel="Arrival rate (customers/hour)")
    ax.grid(alpha=.2); ax.legend(); return _save(fig, path)


def create_signed_accumulation_figure(path: Path) -> Path:
    """Distinguish positive and negative net cash-flow contributions."""
    x = np.linspace(0, 4, 500); y = cash_flow_rate(x)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, y, color="black", linewidth=2); ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(x, 0, y, where=y >= 0, color="tab:green", alpha=.35, label="cash entering (+)")
    ax.fill_between(x, 0, y, where=y < 0, color="tab:red", alpha=.35, label="cash leaving (−)")
    ax.set(title="Signed accumulation: net cash flow", xlabel="Days", ylabel="Net cash-flow rate ($100/day)")
    ax.grid(alpha=.2); ax.legend(); return _save(fig, path)


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 5 accumulation experiment."""
    print("Chapter 5 — Accumulation & Integrals")
    print("Scenario: Harbor Cafe's arrival rate is r(t)=12+6t−t² customers/hour, 0≤t≤6.")
    print("Question: approximately how many customers arrive during those six hours?")
    print("Units: (customers/hour) × hours = customers. A rate is not a customer count.")
    exact = exact_arrivals()
    print("\nRiemann sums (errors compare with the exact model total, 108):")
    print("   n    Δt       left      right   midpoint   |mid error|")
    for n in (3, 6, 12, 24, 48):
        left, right, middle = left_sum(arrival_rate, 0, 6, n), right_sum(arrival_rate, 0, 6, n), midpoint_sum(arrival_rate, 0, 6, n)
        print(f" {n:3d} {6/n:6.3f} {left:10.4f} {right:10.4f} {middle:10.4f} {abs(middle-exact):13.4f}")
    n = 6
    values = {"Left": left_sum(arrival_rate, 0, 6, n), "Right": right_sum(arrival_rate, 0, 6, n), "Midpoint": midpoint_sum(arrival_rate, 0, 6, n), "Trapezoid": trapezoid_rule(arrival_rate, 0, 6, n), "Simpson": simpson_rule(arrival_rate, 0, 6, n)}
    print("\nMethod comparison for n=6:")
    print("  method       approximation   absolute error")
    for name, value in values.items(): print(f"  {name:<11} {value:13.4f} {abs(value-exact):16.4f}")
    scipy_quad, _ = integrate.quad(lambda t: float(arrival_rate(t)), 0, 6)
    scipy_simpson = integrate.simpson(arrival_rate(np.linspace(0, 6, n + 1)), x=np.linspace(0, 6, n + 1))
    print(f"\nExact: R(6)−R(0) = [12t+3t²−t³/3]₀⁶ = {exact:.4f} customers.")
    print(f"SciPy check after our implementations: quad={scipy_quad:.4f}, simpson={scipy_simpson:.4f}.")
    signed = simpson_rule(cash_flow_rate, 0, 4, 4)
    total_magnitude = simpson_rule(lambda t: np.abs(cash_flow_rate(t)), 0, 4, 4)
    print(f"Signed example: c(t)=t−2 gives net={signed:.1f}, but total magnitude={total_magnitude:.1f} hundred-dollar units.")
    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [create_accumulation_figure(destination / "chapter-05-accumulation.png")]
    paths += [create_riemann_figure(destination / f"chapter-05-riemann-{method}.png", method) for method in ("left", "right", "midpoint")]
    paths += [create_refinement_figure(destination / "chapter-05-refinement.png"), create_trapezoid_figure(destination / "chapter-05-trapezoids.png"), create_simpson_figure(destination / "chapter-05-simpson.png"), create_signed_accumulation_figure(destination / "chapter-05-signed-accumulation.png")]
    print("\nInterpretation: the model predicts 108 accumulated customers; it does not promise exactly 108 arrivals.")
    print("Assumptions/limits: arrivals are discrete and uncertain, while the smooth curve is a continuous model. Numerical error and model error differ; extra decimal places cannot repair an unrealistic model.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
