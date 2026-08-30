"""Business rates of change and derivative experiments for Chapter 4."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytics_foundations.datasets import PROJECT_ROOT


def price(q):
    """Return the modeled market price at quantity ``q``."""
    return 30 - 0.05 * np.asarray(q)


def revenue(q):
    """Return revenue, R(q) = 30q - 0.05q²."""
    q = np.asarray(q)
    return 30 * q - 0.05 * q**2


def cost(q):
    """Return cost, C(q) = 200 + 10q."""
    return 200 + 10 * np.asarray(q)


def profit(q):
    """Return profit, P(q) = 20q - 0.05q² - 200."""
    q = np.asarray(q)
    return 20 * q - 0.05 * q**2 - 200


def average_rate_of_change(f: Callable[[float], float], x1: float, x2: float) -> float:
    """Return output change per input unit between two distinct inputs."""
    if x1 == x2:
        raise ValueError("x1 and x2 must differ")
    return float((f(x2) - f(x1)) / (x2 - x1))


def forward_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """Approximate a derivative from ``x`` and a point ``h`` to its right."""
    if h == 0:
        raise ValueError("h must not be zero")
    return float((f(x + h) - f(x)) / h)


def central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """Approximate a derivative using equally spaced points on both sides."""
    if h == 0:
        raise ValueError("h must not be zero")
    return float((f(x + h) - f(x - h)) / (2 * h))


def marginal_revenue(q):
    """Return R'(q), in dollars of revenue per additional unit."""
    return 30 - 0.1 * np.asarray(q)


def marginal_cost(q):
    """Return C'(q), in dollars of cost per additional unit."""
    q = np.asarray(q)
    return np.full_like(q, 10.0, dtype=float) if q.ndim else 10.0


def marginal_profit(q):
    """Return P'(q), in dollars of profit per additional unit."""
    return 20 - 0.1 * np.asarray(q)


def zero_marginal_profit_quantity() -> float:
    """Solve 20 - 0.1q = 0 for this model."""
    return 20 / 0.1


def create_average_rate_figure(output_path: Path) -> Path:
    """Plot the profit curve and the 100-to-110 secant line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q = np.linspace(40, 260, 500)
    q1, q2 = 100.0, 110.0
    slope = average_rate_of_change(profit, q1, q2)
    secant = profit(q1) + slope * (q - q1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q, profit(q), label="Profit P(q)", linewidth=2)
    ax.plot(q, secant, "--", label=f"Secant slope = ${slope:.2f}/unit")
    ax.scatter([q1, q2], [profit(q1), profit(q2)], color="black", zorder=3)
    ax.annotate("interval: 100 to 110", (105, 1290), ha="center")
    ax.set(title="Average profit change across an interval", xlabel="Quantity sold (units)", ylabel="Profit ($)")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    return output_path


def create_secant_tangent_figure(output_path: Path) -> Path:
    """Show wide and narrow secants approaching the tangent at q=100."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q = np.linspace(75, 135, 500); x = 100.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, h, title, color in zip(axes, (20.0, 2.0, None), ("Wide secant (h=20)", "Narrow secant (h=2)", "Tangent (h approaches 0)"), ("tab:orange", "tab:green", "tab:red"), strict=True):
        ax.plot(q, profit(q), linewidth=2, label="P(q)")
        slope = marginal_profit(x) if h is None else forward_difference(profit, x, h)
        ax.plot(q, profit(x) + slope * (q - x), "--", color=color, label=f"slope = {slope:.1f}")
        ax.scatter([x], [profit(x)], color="black", zorder=3)
        if h is not None: ax.scatter([x + h], [profit(x + h)], color=color, zorder=3)
        ax.set(title=title, xlabel="Quantity q"); ax.grid(alpha=0.2); ax.legend()
    axes[0].set_ylabel("Profit ($)")
    fig.suptitle("Secant slopes approach the tangent slope at q=100")
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    return output_path


def create_profit_marginal_figure(output_path: Path) -> Path:
    """Relate the profit maximum to positive, zero, and negative marginal profit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q = np.linspace(0, 300, 600); optimum = zero_marginal_profit_quantity()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(q, profit(q), linewidth=2); top.scatter([optimum], [profit(optimum)], color="black", zorder=3)
    top.annotate(f"maximum profit = ${profit(optimum):,.0f}", (optimum, profit(optimum)), xytext=(215, 1500), arrowprops={"arrowstyle": "->"})
    top.set(ylabel="Profit P(q) ($)", title="Profit rises, becomes flat, then falls"); top.grid(alpha=0.25)
    mp = marginal_profit(q)
    bottom.plot(q, mp, color="tab:purple", linewidth=2); bottom.axhline(0, color="black", linewidth=1)
    bottom.fill_between(q, 0, mp, where=mp > 0, alpha=0.2, color="tab:green", label="P'(q) > 0: profit rising")
    bottom.fill_between(q, 0, mp, where=mp < 0, alpha=0.2, color="tab:red", label="P'(q) < 0: profit falling")
    bottom.scatter([optimum], [0], color="black", zorder=3, label="P'(200) = 0")
    bottom.set(xlabel="Quantity sold (units)", ylabel="Marginal profit ($/unit)"); bottom.grid(alpha=0.25); bottom.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    return output_path


def run(output_dir: Path | None = None) -> int:
    """Run the Chapter 4 derivative experiment."""
    print("Chapter 4 — Change & Derivatives")
    print("Scenario: demand p(q)=30-0.05q; R(q)=30q-0.05q²; C(q)=200+10q.")
    print("Question: how quickly does profit change when quantity changes?\n")
    for q in (100, 110, 150, 200): print(f"  P({q:3}) = ${profit(q):,.2f}")
    change = float(profit(110) - profit(100)); rate = average_rate_of_change(profit, 100, 110)
    print(f"\n100 → 110: ΔP=${change:,.2f}, Δq=10, average rate=${rate:.2f} per unit.")
    print("\nShrinking h at q=100 (forward difference, $ profit per unit):")
    print("       h       [P(100+h)-P(100)]/h")
    for h in (20, 10, 5, 1, .1, .01): print(f"  {h:7g} {forward_difference(profit, 100, h):25.4f}")
    analytical = float(marginal_profit(100))
    print(f"  limit → P'(100) = {analytical:.2f} dollars of profit per unit")
    print("\nAnalytical versus numerical derivative at q=100:")
    print(f"  analytical={analytical:.4f}; forward(h=.01)={forward_difference(profit, 100, .01):.4f}; central(h=.01)={central_difference(profit, 100, .01):.4f}")
    print(f"\nAt q=100: marginal revenue=${marginal_revenue(100):.2f}/unit, marginal cost=${marginal_cost(100):.2f}/unit, marginal profit=${marginal_profit(100):.2f}/unit.")
    optimum = zero_marginal_profit_quantity()
    print(f"P'(q)=0 at q={optimum:.0f}; total profit there is ${profit(optimum):,.2f}, not $0.")
    print(f"For a whole unit at q=100, P(101)-P(100)=${profit(101)-profit(100):.2f}, versus P'(100)=${analytical:.2f}/unit.")
    destination = output_dir or PROJECT_ROOT / "figures"
    paths = [create_average_rate_figure(destination / "chapter-04-average-rate.png"), create_secant_tangent_figure(destination / "chapter-04-secant-to-tangent.png"), create_profit_marginal_figure(destination / "chapter-04-profit-and-marginal-profit.png")]
    print("\nInterpretation: positive marginal profit supports a small increase; zero marks a locally flat profit curve; negative marginal profit warns that more quantity lowers modeled profit.")
    print("Assumptions/limits: continuous quantities, certain demand, fixed cost structure, and no capacity, inventory, labor, or competitor response. Operational decisions require those realities.")
    print("Figures saved to: " + ", ".join(map(str, paths)))
    return 0
