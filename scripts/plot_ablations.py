#!/usr/bin/env python3
"""Reproduce the paper's ablation figures from the supplied CSV data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def axes():
    fig, ax = plt.subplots(figsize=(3.4, 2.65), layout="constrained")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "Figures" / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


SERIES = (
    ("static_mean_pct", "Static mean", "#D55E00", "s", "--"),
    ("static_minimum_pct", "Static minimum", "#009E73", "^", "-."),
    ("random_success_pct", "vs. Random", "#0072B2", "o", "-"),
    ("ibr_success_pct", "vs. Q-IBR", "#CC79A7", "D", ":"),
    ("other_proposed_success_pct", "vs. Other proposed", "#666666", "v", (0, (4, 1, 1, 1))),
)


def plot_series(ax, x, data):
    for key, label, color, marker, style in SERIES:
        ax.plot(x, data[key], label=label, color=color, marker=marker,
                linestyle=style, linewidth=1.5, markersize=4)
    ax.legend(loc="lower right", frameon=True, edgecolor="0.85",
              labelspacing=0.25, borderpad=0.35, handlelength=2.1)


def main():
    data = np.genfromtxt(ROOT / "data/ablation_training_size.csv", delimiter=",", names=True)
    fig, ax = axes()
    x = data["collection_size_rallies"]
    plot_series(ax, x, data)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.minorticks_off()
    ax.set_xlabel("Collection size (rallies)")
    save(fig, "ablation_training_size")

    data = np.genfromtxt(ROOT / "data/ablation_discount_factor.csv", delimiter=",", names=True)
    fig, ax = axes()
    x = data["discount_factor"]
    plot_series(ax, x, data)
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_xticks(x)
    ax.set_xlim(0.48, 1.02)
    save(fig, "ablation_discount_factor")


if __name__ == "__main__":
    main()
