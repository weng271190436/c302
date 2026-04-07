#!/usr/bin/env python3
"""
C. elegans Connectome Visualization
Visualizes the 302 neurons with their connections, colored by neuron type.

Usage:
    python visualize_connectome.py
    python visualize_connectome.py --output connectome.png
    python visualize_connectome.py --layout spring  # or kamada_kawai, circular, shell
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Neuron type classifications based on WormAtlas
# Source: https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html

SENSORY_NEURONS = {
    # Amphid sensory neurons (chemosensory, thermosensory)
    'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR',
    'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASHL', 'ASHR',
    'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR',
    'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR',
    # Phasmid sensory neurons
    'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL', 'PHCR',
    # Touch receptor neurons
    'ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM',
    # Anterior deirid neurons
    'ADEL', 'ADER',
    # Posterior deirid
    'PDEL', 'PDER',
    # CEP neurons (dopaminergic mechanosensory)
    'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR',
    # IL neurons (inner labial)
    'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL', 'IL1VR',
    'IL2DL', 'IL2DR', 'IL2L', 'IL2R', 'IL2VL', 'IL2VR',
    # OL neurons (outer labial)
    'OLLL', 'OLLR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR',
    # URX/URY (oxygen sensing)
    'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR',
    # BAG (oxygen/CO2 sensing)
    'BAGL', 'BAGR',
    # AQR/PQR (oxygen sensing)
    'AQR', 'PQR',
    # FLP/PVD (polymodal nociceptive)
    'FLPL', 'FLPR', 'PVDL', 'PVDR',
    # Other sensory
    'ALNL', 'ALNR', 'PLNL', 'PLNR',
}

MOTOR_NEURONS = {
    # Ventral cord motor neurons - A class (backward)
    'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9',
    'VA1', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11', 'VA12',
    # Ventral cord motor neurons - B class (forward)
    'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7',
    'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10', 'VB11',
    # Ventral cord motor neurons - D class (inhibitory)
    'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6',
    'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13',
    # AS motor neurons
    'AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11',
    # VC motor neurons (egg-laying)
    'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6',
    # Head motor neurons
    'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMDDL', 'RMDDR',
    'RMEL', 'RMER', 'RMED', 'RMEV',
    'RMFL', 'RMFR',
    'RMGL', 'RMGR',
    'RMHL', 'RMHR',
    'RIVL', 'RIVR',
    'RIML', 'RIMR',
    # SMD/SMB motor neurons
    'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR',
    'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR',
    # SAA/SAB motor neurons
    'SAADL', 'SAADR', 'SAAVL', 'SAAVR',
    'SABDL', 'SABDR', 'SABVL', 'SABVR',
    # SIA/SIB motor neurons
    'SIADL', 'SIADR', 'SIAVL', 'SIAVR',
    'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR',
    # URA/URB motor neurons
    'URADL', 'URADR', 'URAVL', 'URAVR',
    'URBL', 'URBR',
    # Other motor neurons
    'DVB', 'DVA', 'PDA', 'PDB', 'PVT',
    'RID', 'AVL',
    # HSN (egg-laying)
    'HSNL', 'HSNR',
}

# Everything else is an interneuron
# Notable interneurons for locomotion command:
COMMAND_INTERNEURONS = {
    'AVAL', 'AVAR',  # Backward command
    'AVBL', 'AVBR',  # Forward command
    'AVDL', 'AVDR',  # Backward
    'AVEL', 'AVER',  # Backward
    'PVCL', 'PVCR',  # Forward command
}


def get_neuron_type(neuron_name):
    """Classify a neuron as sensory, motor, command interneuron, or interneuron."""
    name = neuron_name.strip().upper()
    if name in SENSORY_NEURONS:
        return 'sensory'
    elif name in MOTOR_NEURONS:
        return 'motor'
    elif name in COMMAND_INTERNEURONS:
        return 'command'
    else:
        return 'interneuron'


def load_connectome(data_dir):
    """Load connectome from the c302 data files."""
    edgelist_path = data_dir / 'herm_full_edgelist.csv'
    
    if not edgelist_path.exists():
        raise FileNotFoundError(f"Could not find {edgelist_path}")
    
    # Read edge list
    df = pd.read_csv(edgelist_path)
    
    # Clean column names and data
    df.columns = [c.strip() for c in df.columns]
    df['Source'] = df['Source'].str.strip()
    df['Target'] = df['Target'].str.strip()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with attributes
    for _, row in df.iterrows():
        source = row['Source'].upper()
        target = row['Target'].upper()
        weight = row.get('Weight', 1)
        conn_type = row.get('Type', 'chemical')
        
        # Skip muscle connections for now (lowercase names like 'pm1vl')
        if not source[0].isupper() or not target[0].isupper():
            continue
            
        if G.has_edge(source, target):
            # Add to existing edge weight
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight, type=conn_type)
    
    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['type'] = get_neuron_type(node)
    
    return G


def visualize_connectome(G, layout='kamada_kawai', output=None, show_labels=False):
    """Create a visualization of the connectome."""
    
    # Color mapping
    type_colors = {
        'sensory': '#2ecc71',      # Green
        'motor': '#e74c3c',        # Red  
        'command': '#f39c12',      # Orange
        'interneuron': '#3498db',  # Blue
    }
    
    # Get node colors
    node_colors = [type_colors.get(G.nodes[n].get('type', 'interneuron'), '#3498db') 
                   for n in G.nodes()]
    
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 400 * (degrees[n] / max_degree) for n in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Calculate layout
    print(f"Calculating {layout} layout for {G.number_of_nodes()} neurons...")
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        # Arrange by type in shells
        shells = [
            [n for n in G.nodes() if G.nodes[n].get('type') == 'sensory'],
            [n for n in G.nodes() if G.nodes[n].get('type') == 'interneuron'],
            [n for n in G.nodes() if G.nodes[n].get('type') == 'command'],
            [n for n in G.nodes() if G.nodes[n].get('type') == 'motor'],
        ]
        shells = [s for s in shells if s]  # Remove empty shells
        pos = nx.shell_layout(G, shells)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw edges (very light, as there are many)
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.1 + 0.3 * (w / max_weight) for w in edge_weights]
    
    # Draw chemical synapses
    chemical_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'chemical']
    gap_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'chemical']
    
    nx.draw_networkx_edges(G, pos, edgelist=chemical_edges, 
                           alpha=0.15, edge_color='gray', 
                           arrows=True, arrowsize=5,
                           connectionstyle="arc3,rad=0.1",
                           ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=gap_edges,
                           alpha=0.1, edge_color='purple',
                           style='dashed', arrows=False,
                           ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Optionally draw labels
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c=type_colors['sensory'], s=100, label=f'Sensory ({len([n for n in G.nodes() if G.nodes[n].get("type") == "sensory"])})'),
        plt.scatter([], [], c=type_colors['interneuron'], s=100, label=f'Interneuron ({len([n for n in G.nodes() if G.nodes[n].get("type") == "interneuron"])})'),
        plt.scatter([], [], c=type_colors['command'], s=100, label=f'Command ({len([n for n in G.nodes() if G.nodes[n].get("type") == "command"])})'),
        plt.scatter([], [], c=type_colors['motor'], s=100, label=f'Motor ({len([n for n in G.nodes() if G.nodes[n].get("type") == "motor"])})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Title and info
    ax.set_title(f'C. elegans Connectome\n{G.number_of_nodes()} neurons, {G.number_of_edges()} connections', 
                 fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output}")
    
    plt.show()
    
    return G, pos


def print_stats(G):
    """Print statistics about the connectome."""
    print("\n=== C. elegans Connectome Statistics ===\n")
    
    print(f"Total neurons: {G.number_of_nodes()}")
    print(f"Total connections: {G.number_of_edges()}")
    print()
    
    # Count by type
    type_counts = {}
    for node in G.nodes():
        t = G.nodes[node].get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("Neurons by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()
    
    # Connection statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    print("Most connected neurons (total degree):")
    total_degrees = {n: in_degrees[n] + out_degrees[n] for n in G.nodes()}
    for n, d in sorted(total_degrees.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n}: {d} ({G.nodes[n].get('type')})")
    print()
    
    print("Highest out-degree (most outputs):")
    for n, d in sorted(out_degrees.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n}: {d} ({G.nodes[n].get('type')})")
    print()
    
    print("Highest in-degree (most inputs):")
    for n, d in sorted(in_degrees.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n}: {d} ({G.nodes[n].get('type')})")


def main():
    parser = argparse.ArgumentParser(description='Visualize C. elegans connectome')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to c302 data directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (e.g., connectome.png)')
    parser.add_argument('--layout', type=str, default='kamada_kawai',
                        choices=['spring', 'kamada_kawai', 'circular', 'shell', 'spectral'],
                        help='Graph layout algorithm')
    parser.add_argument('--labels', action='store_true',
                        help='Show neuron labels')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, no visualization')
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try common locations
        possible_paths = [
            Path(__file__).parent / 'c302' / 'data',
            Path(__file__).parent / 'data',
            Path.home() / 'repos' / 'weng-c302' / 'c302' / 'data',
            Path.home() / 'repos' / 'c302' / 'c302' / 'data',
        ]
        data_dir = None
        for p in possible_paths:
            if p.exists():
                data_dir = p
                break
        
        if data_dir is None:
            print("Could not find c302 data directory. Please specify with --data-dir")
            return 1
    
    print(f"Loading connectome from {data_dir}...")
    G = load_connectome(data_dir)
    
    print_stats(G)
    
    if not args.stats_only:
        visualize_connectome(G, layout=args.layout, output=args.output, 
                            show_labels=args.labels)
    
    return 0


if __name__ == '__main__':
    exit(main())
