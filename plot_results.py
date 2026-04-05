#!/usr/bin/env python
"""
Plot simulation results from c302 .dat files with proper zoom controls.

Usage:
    python plot_results.py c302_C_IClamp.dat
    python plot_results.py c302_C_IClamp.dat --xlim 4000 5000
    python plot_results.py c302_C_IClamp.dat --cols 1 2 3
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def get_column_names(datfile):
    """Try to extract column names from corresponding LEMS XML file."""
    # Guess the LEMS file name from dat file
    basename = os.path.basename(datfile)
    # c302_C_IClamp.dat -> LEMS_c302_C_IClamp.xml
    if basename.endswith('.activity.dat'):
        lems_name = 'LEMS_' + basename.replace('.activity.dat', '.xml')
    elif basename.endswith('.muscles.dat'):
        lems_name = 'LEMS_' + basename.replace('.muscles.dat', '.xml')
    elif basename.endswith('.muscles.activity.dat'):
        lems_name = 'LEMS_' + basename.replace('.muscles.activity.dat', '.xml')
    else:
        lems_name = 'LEMS_' + basename.replace('.dat', '.xml')
    
    # Look in examples/ directory
    lems_paths = [
        os.path.join(os.path.dirname(datfile), lems_name),
        os.path.join(os.path.dirname(datfile), 'examples', lems_name),
        os.path.join('examples', lems_name),
    ]
    
    for lems_path in lems_paths:
        if os.path.exists(lems_path):
            try:
                with open(lems_path, 'r') as f:
                    content = f.read()
                # Extract OutputColumn ids
                # <OutputColumn id="ADAL_v" quantity="ADAL/0/GenericNeuronCell/v"/>
                pattern = r'<OutputColumn\s+id="([^"]+)"'
                matches = re.findall(pattern, content)
                if matches:
                    return ['time'] + matches
            except Exception:
                pass
    
    return None


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

    # Try to get column names
    col_names = get_column_names(args.datfile)
    
    # Determine which columns to plot
    if args.cols:
        cols = args.cols
    else:
        cols = list(range(1, data.shape[1]))  # all except time column

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cols:
        if col_names and col < len(col_names):
            label = col_names[col]
        else:
            label = f"col {col}"
        ax.plot(t, data[:, col], label=label, linewidth=1.5)

    # Apply limits
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)

    # Labels
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Membrane Potential (V)", fontsize=12)
    if args.title:
        ax.set_title(args.title, fontsize=14)
    else:
        ax.set_title(os.path.basename(args.datfile), fontsize=14)

    if not args.no_legend and len(cols) <= 10:
        ax.legend(loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
