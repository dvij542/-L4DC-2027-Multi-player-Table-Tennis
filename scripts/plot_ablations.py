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


def main():
    data = np.genfromtxt(ROOT / "data/ablation_training_size.csv", delimiter=",", names=True)
    fig, ax = axes()
    x = data["training_size"]
    for key, label, color, marker, style in (
        ("random_success_pct", "Random opponent", "#0072B2", "o", "-"),
        ("static_mean_pct", "Static mean", "#D55E00", "s", "--"),
        ("average_static_minimum_pct", "Average static minimum", "#009E73", "^", "-."),
    ):
        ax.plot(x, data[key], label=label, color=color, marker=marker, linestyle=style, linewidth=1.8, markersize=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.minorticks_off()
    ax.set_xlabel("Training-set size")
    ax.legend(loc="lower right", frameon=True, edgecolor="0.85")
    save(fig, "ablation_training_size")

    data = np.genfromtxt(ROOT / "data/ablation_discount_factor.csv", delimiter=",", names=True)
    fig, ax = axes()
    x = data["discount_factor"]
    ax.plot(x, data["random_success_pct"], label="Random opponent", color="#0072B2", marker="o", linewidth=1.8, markersize=4)
    ax.plot(x, data["static_mean_pct"], label="15 fixed opponents (mean)", color="#D55E00", marker="s", linestyle="--", linewidth=1.8, markersize=4)
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.48, 1.02)
    ax.legend(loc="lower right", frameon=True, edgecolor="0.85")
    save(fig, "ablation_discount_factor")


if __name__ == "__main__":
    main()
