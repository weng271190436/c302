#!/usr/bin/env python3
"""
Visualize signal flow from a specific neuron through the C. elegans connectome.
Shows all downstream neurons reachable from the source.

Usage:
    python visualize_from_neuron.py AWCL
    python visualize_from_neuron.py AWCL AWCR --depth 3
    python visualize_from_neuron.py PLML --output touch_response.png
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

# Import neuron type classifications
from visualize_connectome import (
    SENSORY_NEURONS, MOTOR_NEURONS, COMMAND_INTERNEURONS, 
    get_neuron_type, load_connectome
)


def get_downstream_subgraph(G, sources, max_depth=4):
    """Get all neurons reachable from source neurons within max_depth hops."""
    
    # BFS to find all reachable nodes with their distances
    distances = {}
    for source in sources:
        if source not in G:
            print(f"Warning: {source} not found in connectome")
            continue
        distances[source] = 0
    
    visited = set(sources)
    frontier = list(sources)
    
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node in frontier:
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    
    # Create subgraph
    subgraph = G.subgraph(visited).copy()
    
    # Add distance attribute
    for node in subgraph.nodes():
        subgraph.nodes[node]['distance'] = distances.get(node, 0)
    
    return subgraph, distances


def visualize_from_neuron(G, sources, max_depth=4, output=None, show_labels=True):
    """Visualize signal propagation from source neurons."""
    
    subgraph, distances = get_downstream_subgraph(G, sources, max_depth)
    
    print(f"\n=== Signal Flow from {', '.join(sources)} ===\n")
    print(f"Reachable neurons: {subgraph.number_of_nodes()}")
    print(f"Connections: {subgraph.number_of_edges()}")
    print()
    
    # Count by depth
    depth_counts = defaultdict(list)
    for node, dist in distances.items():
        depth_counts[dist].append(node)
    
    print("Neurons by distance from source:")
    for depth in sorted(depth_counts.keys()):
        neurons = depth_counts[depth]
        types = defaultdict(int)
        for n in neurons:
            types[get_neuron_type(n)] += 1
        type_str = ", ".join(f"{t}: {c}" for t, c in sorted(types.items()))
        print(f"  Depth {depth}: {len(neurons)} neurons ({type_str})")
        if depth <= 2 and len(neurons) <= 20:
            print(f"    {', '.join(sorted(neurons))}")
    
    # Color by distance from source
    cmap = plt.cm.YlOrRd  # Yellow -> Orange -> Red
    max_dist = max(distances.values()) if distances else 1
    
    node_colors = []
    for n in subgraph.nodes():
        dist = distances.get(n, 0)
        if n in sources:
            node_colors.append('#00ff00')  # Green for source
        else:
            node_colors.append(cmap(dist / max_dist))
    
    # Node sizes based on out-degree (influence)
    out_degrees = dict(subgraph.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    node_sizes = [200 + 600 * (out_degrees[n] / max_out) for n in subgraph.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Use hierarchical layout based on distance
    # Create layers for each distance
    layers = defaultdict(list)
    for node, dist in distances.items():
        layers[dist].append(node)
    
    # Position nodes in layers
    pos = {}
    for dist, nodes in layers.items():
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            # Spread nodes horizontally, stack by distance vertically
            x = (i - n_nodes / 2) * 0.5
            y = -dist  # Negative so source is at top
            pos[node] = (x, y)
    
    # Draw edges
    edge_colors = []
    for u, v in subgraph.edges():
        if u in sources:
            edge_colors.append('#00aa00')  # Green from source
        else:
            edge_colors.append('#888888')
    
    nx.draw_networkx_edges(subgraph, pos, 
                           edge_color=edge_colors,
                           alpha=0.4, 
                           arrows=True, 
                           arrowsize=10,
                           connectionstyle="arc3,rad=0.1",
                           ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                           node_color=node_colors,
                           node_size=node_sizes, 
                           alpha=0.9,
                           ax=ax)
    
    # Draw labels
    if show_labels:
        # Only label important nodes to reduce clutter
        labels = {}
        for n in subgraph.nodes():
            dist = distances.get(n, 0)
            if dist <= 2 or n in sources or out_degrees[n] > 5:
                labels[n] = n
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=7, ax=ax)
    
    # Add legend for distance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00ff00', label=f'Source: {", ".join(sources)}'),
        Patch(facecolor=cmap(0.25), label='1 hop'),
        Patch(facecolor=cmap(0.5), label='2 hops'),
        Patch(facecolor=cmap(0.75), label='3 hops'),
        Patch(facecolor=cmap(1.0), label=f'4+ hops'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Title
    ax.set_title(f'Signal Flow from {", ".join(sources)}\n'
                 f'{subgraph.number_of_nodes()} neurons reachable in {max_depth} hops', 
                 fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved to {output}")
    
    plt.show()
    
    # Print interesting paths
    print("\n=== Key Signal Paths ===\n")
    
    # Find paths to motor neurons
    motor_targets = [n for n in subgraph.nodes() if get_neuron_type(n) == 'motor']
    if motor_targets:
        print(f"Motor neurons reached: {len(motor_targets)}")
        for motor in sorted(motor_targets)[:5]:
            for source in sources:
                if source in subgraph and motor in subgraph:
                    try:
                        path = nx.shortest_path(subgraph, source, motor)
                        print(f"  {source} → {motor}: {' → '.join(path)}")
                    except nx.NetworkXNoPath:
                        pass
    
    # Find paths to command interneurons
    command_targets = [n for n in subgraph.nodes() if get_neuron_type(n) == 'command']
    if command_targets:
        print(f"\nCommand interneurons reached: {command_targets}")
        for cmd in command_targets:
            for source in sources:
                if source in subgraph and cmd in subgraph:
                    try:
                        path = nx.shortest_path(subgraph, source, cmd)
                        print(f"  {source} → {cmd}: {' → '.join(path)}")
                    except nx.NetworkXNoPath:
                        pass
    
    return subgraph


def main():
    parser = argparse.ArgumentParser(description='Visualize signal flow from neurons')
    parser.add_argument('neurons', nargs='+', help='Source neuron(s) (e.g., AWCL AWCR)')
    parser.add_argument('--depth', type=int, default=4, help='Max hops from source')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file')
    parser.add_argument('--no-labels', action='store_true', help='Hide neuron labels')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to c302 data')
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / 'c302' / 'data'
    
    print(f"Loading connectome from {data_dir}...")
    G = load_connectome(data_dir)
    
    # Normalize neuron names
    sources = [n.upper().strip() for n in args.neurons]
    
    visualize_from_neuron(G, sources, max_depth=args.depth, 
                         output=args.output, show_labels=not args.no_labels)


if __name__ == '__main__':
    main()
