#!/usr/bin/env python
"""
Plot simulation results from c302 .dat files with proper zoom controls.

Usage:
    python plot_results.py c302_C_IClamp.dat
    python plot_results.py c302_C_IClamp.dat --xlim 4000 5000
    python plot_results.py c302_C_IClamp.dat --cols 1 2 3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot c302 simulation results")
    parser.add_argument("datfile", help="Path to .dat file")
    parser.add_argument("--xlim", nargs=2, type=float, help="X-axis limits (ms)")
    parser.add_argument("--ylim", nargs=2, type=float, help="Y-axis limits")
    parser.add_argument("--cols", nargs="+", type=int, help="Column indices to plot (default: all)")
    parser.add_argument("--no-legend", action="store_true", help="Hide legend")
    parser.add_argument("--title", type=str, help="Plot title")
    args = parser.parse_args()

    # Load data
    data = np.loadtxt(args.datfile)
    t = data[:, 0] * 1000  # convert s to ms

    # Determine which columns to plot
    if args.cols:
        cols = args.cols
    else:
        cols = list(range(1, data.shape[1]))  # all except time column

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cols:
        ax.plot(t, data[:, col], label=f"col {col}")

    # Apply limits
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)

    # Labels
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(args.datfile)

    if not args.no_legend and len(cols) <= 10:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
