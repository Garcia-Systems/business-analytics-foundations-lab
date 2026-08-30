"""Executable revenue, cost, and profit models for Chapter 1."""

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from analytics_foundations.datasets import PROJECT_ROOT

PRICE = 12
FIXED_COST = 300
VARIABLE_COST = 5


def revenue(quantity: ArrayLike) -> NDArray[np.number] | float:
    """Return revenue at ``quantity`` under the chapter's $12 unit price."""

    return PRICE * quantity


def cost(quantity: ArrayLike) -> NDArray[np.number] | float:
    """Return total cost at ``quantity`` under the chapter's cost model."""

    return FIXED_COST + VARIABLE_COST * quantity


def profit(quantity: ArrayLike) -> NDArray[np.number] | float:
    """Return revenue less cost at ``quantity``."""

    return revenue(quantity) - cost(quantity)


def break_even_quantity(
    price: float = PRICE,
    fixed_cost: float = FIXED_COST,
    variable_cost: float = VARIABLE_COST,
) -> float:
    """Return the mathematical quantity where revenue equals cost.

    A price no greater than variable cost never recovers a positive fixed cost,
    so this intentionally simple model has no finite break-even quantity then.
    """

    contribution_margin = price - variable_cost
    if contribution_margin <= 0:
        raise ValueError("price must be greater than variable cost")
    if fixed_cost < 0:
        raise ValueError("fixed cost cannot be negative")
    return fixed_cost / contribution_margin


def operational_break_even_quantity(
    price: float = PRICE,
    fixed_cost: float = FIXED_COST,
    variable_cost: float = VARIABLE_COST,
) -> int:
    """Return the first whole-unit quantity with nonnegative profit."""

    return ceil(break_even_quantity(price, fixed_cost, variable_cost))


def create_revenue_cost_figure(quantities: ArrayLike, output_path: Path) -> Path:
    """Plot revenue and cost, including their mathematical intersection."""

    quantities = np.asarray(quantities)
    break_even = break_even_quantity()
    break_even_value = float(revenue(break_even))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(quantities, revenue(quantities), label="Revenue: R(q) = 12q")
    axis.plot(quantities, cost(quantities), label="Cost: C(q) = 300 + 5q")
    axis.scatter([break_even], [break_even_value], color="black", zorder=3)
    axis.annotate(
        f"Break-even\n({break_even:.1f}, ${break_even_value:.2f})",
        xy=(break_even, break_even_value),
        xytext=(12, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    axis.scatter([0], [FIXED_COST], color="tab:orange", zorder=3)
    axis.annotate(
        "Fixed cost = $300",
        xy=(0, FIXED_COST),
        xytext=(25, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:orange"},
    )
    axis.set(
        title="Revenue and cost models",
        xlabel="Quantity sold (units)",
        ylabel="Dollars ($)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def create_profit_figure(quantities: ArrayLike, output_path: Path) -> Path:
    """Plot profit and mark where it crosses zero."""

    quantities = np.asarray(quantities)
    break_even = break_even_quantity()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(quantities, profit(quantities), color="tab:green", label="P(q) = 7q - 300")
    axis.axhline(0, color="black", linewidth=1)
    axis.scatter([break_even], [0], color="black", zorder=3)
    axis.annotate(
        f"Profit = $0 at q = {break_even:.1f}",
        xy=(break_even, 0),
        xytext=(12, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    axis.set(
        title="Profit model",
        xlabel="Quantity sold (units)",
        ylabel="Profit ($)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run(output_dir: Path | None = None) -> int:
    """Run and narrate the Chapter 1 functions experiment."""

    print("Chapter 1 — Functions Become Models")
    print("Scenario: A pop-up bakery sells treat boxes for $12 each.\n")
    print("Models:")
    print("  Revenue: R(q) = 12q")
    print("  Cost:    C(q) = 300 + 5q")
    print("  Profit:  P(q) = R(q) - C(q) = 7q - 300")
    print("\nSelected model evaluations:")
    print("  Quantity   Revenue      Cost    Profit")
    for quantity in (0, 20, 40, 43, 60):
        print(
            f"  {quantity:8d}  ${revenue(quantity):8.2f}  "
            f"${cost(quantity):8.2f}  ${profit(quantity):8.2f}"
        )

    mathematical = break_even_quantity()
    operational = operational_break_even_quantity()
    print(f"\nMathematical break-even: {mathematical:.2f} units")
    print(f"Operational break-even:  {operational} whole units")

    quantities = np.arange(0, 101)
    destination = output_dir or PROJECT_ROOT / "figures"
    revenue_cost_path = create_revenue_cost_figure(
        quantities, destination / "chapter-01-revenue-and-cost.png"
    )
    profit_path = create_profit_figure(
        quantities, destination / "chapter-01-profit.png"
    )
    print(
        "\nInterpretation: The model predicts break-even at approximately "
        f"{mathematical:.1f} units. Because boxes are sold whole, at least "
        f"{operational} boxes must be sold for revenue to exceed cost."
    )
    print("Assumptions:")
    print("  - Price and variable cost per box remain constant.")
    print("  - Fixed cost remains $300, and every box is treated identically.")
    print("  - Every box made is sold; demand and capacity are not modeled.")
    print("Models simplify reality so that we can calculate and reason with it.")
    print(f"Figures saved to: {revenue_cost_path} and {profit_path}")
    return 0
