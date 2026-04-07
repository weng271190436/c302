#!/usr/bin/env python3
"""
Interactive C. elegans Connectome Explorer
Creates an HTML file with draggable, clickable neurons.

Usage:
    python interactive_connectome.py AWCL AWCR --depth 3
    python interactive_connectome.py --full  # Full connectome (slow)
"""

import networkx as nx
from pyvis.network import Network
import pandas as pd
from pathlib import Path
import argparse

# Neuron classifications and descriptions
NEURON_INFO = {
    # Sensory - Chemosensory
    'AWCL': {'type': 'sensory', 'desc': 'Odor sensor (ON type) - detects benzaldehyde, isoamyl alcohol. Asymmetric with AWCR.'},
    'AWCR': {'type': 'sensory', 'desc': 'Odor sensor (OFF type) - detects butanone, 2,3-pentanedione. Asymmetric with AWCL.'},
    'ASEL': {'type': 'sensory', 'desc': 'Salt sensor (left) - attracts to Na+, Li+. Asymmetric pair.'},
    'ASER': {'type': 'sensory', 'desc': 'Salt sensor (right) - attracts to Cl-, K+. Asymmetric pair.'},
    'ASHL': {'type': 'sensory', 'desc': 'Polymodal nociceptor - nose touch, osmotic stress, bitter chemicals.'},
    'ASHR': {'type': 'sensory', 'desc': 'Polymodal nociceptor - nose touch, osmotic stress, bitter chemicals.'},
    'ADFL': {'type': 'sensory', 'desc': 'Detects dauer pheromone and some odors. Modulates lifespan.'},
    'ADFR': {'type': 'sensory', 'desc': 'Detects dauer pheromone and some odors. Modulates lifespan.'},
    
    # Sensory - Thermosensory
    'AFDL': {'type': 'sensory', 'desc': 'Thermosensor - incredible 0.01°C precision! Remembers cultivation temperature.'},
    'AFDR': {'type': 'sensory', 'desc': 'Thermosensor - incredible 0.01°C precision! Remembers cultivation temperature.'},
    
    # Sensory - Mechanosensory
    'ALML': {'type': 'sensory', 'desc': 'Gentle touch receptor (anterior body). Key for tap withdrawal.'},
    'ALMR': {'type': 'sensory', 'desc': 'Gentle touch receptor (anterior body). Key for tap withdrawal.'},
    'PLML': {'type': 'sensory', 'desc': 'Gentle touch receptor (posterior body). Triggers forward movement.'},
    'PLMR': {'type': 'sensory', 'desc': 'Gentle touch receptor (posterior body). Triggers forward movement.'},
    'AVM': {'type': 'sensory', 'desc': 'Gentle touch receptor (mid-body ventral). Part of touch circuit.'},
    'PVM': {'type': 'sensory', 'desc': 'Gentle touch receptor (mid-body). Develops post-embryonically.'},
    
    # Sensory - Oxygen
    'URXL': {'type': 'sensory', 'desc': 'Oxygen sensor - worm prefers ~8% O2, avoids high and low.'},
    'URXR': {'type': 'sensory', 'desc': 'Oxygen sensor - worm prefers ~8% O2, avoids high and low.'},
    'BAGL': {'type': 'sensory', 'desc': 'Oxygen/CO2 sensor - also responds to carbon dioxide.'},
    'BAGR': {'type': 'sensory', 'desc': 'Oxygen/CO2 sensor - also responds to carbon dioxide.'},
    
    # Key Interneurons
    'AIBL': {'type': 'interneuron', 'desc': 'Sensory integration hub - receives AWC, ASE, ASH. Key for chemotaxis.'},
    'AIBR': {'type': 'interneuron', 'desc': 'Sensory integration hub - receives AWC, ASE, ASH. Key for chemotaxis.'},
    'AIYL': {'type': 'interneuron', 'desc': 'Thermotaxis interneuron - processes AFD temperature signals.'},
    'AIYR': {'type': 'interneuron', 'desc': 'Thermotaxis interneuron - processes AFD temperature signals.'},
    'AIZL': {'type': 'interneuron', 'desc': 'Sensory processing - integrates multiple sensory modalities.'},
    'AIZR': {'type': 'interneuron', 'desc': 'Sensory processing - integrates multiple sensory modalities.'},
    'RIAL': {'type': 'interneuron', 'desc': 'Head movement control - integrates sensory for head steering.'},
    'RIAR': {'type': 'interneuron', 'desc': 'Head movement control - integrates sensory for head steering.'},
    'RIBL': {'type': 'interneuron', 'desc': 'Processes mechanosensory input for locomotion.'},
    'RIBR': {'type': 'interneuron', 'desc': 'Processes mechanosensory input for locomotion.'},
    
    # Command Interneurons
    'AVAL': {'type': 'command', 'desc': '⭐ BACKWARD COMMAND - activates A-class motor neurons for reverse.'},
    'AVAR': {'type': 'command', 'desc': '⭐ BACKWARD COMMAND - activates A-class motor neurons for reverse.'},
    'AVBL': {'type': 'command', 'desc': '⭐ FORWARD COMMAND - activates B-class motor neurons for forward.'},
    'AVBR': {'type': 'command', 'desc': '⭐ FORWARD COMMAND - activates B-class motor neurons for forward.'},
    'PVCL': {'type': 'command', 'desc': '⭐ FORWARD COMMAND - posterior touch triggers forward via PVC.'},
    'PVCR': {'type': 'command', 'desc': '⭐ FORWARD COMMAND - posterior touch triggers forward via PVC.'},
    'AVDL': {'type': 'command', 'desc': 'Backward movement - responds to anterior touch.'},
    'AVDR': {'type': 'command', 'desc': 'Backward movement - responds to anterior touch.'},
    'AVEL': {'type': 'command', 'desc': 'Backward movement - part of escape response.'},
    'AVER': {'type': 'command', 'desc': 'Backward movement - part of escape response.'},
    
    # Motor Neurons
    'DA1': {'type': 'motor', 'desc': 'Dorsal A-class motor - backward locomotion, dorsal muscles.'},
    'VA1': {'type': 'motor', 'desc': 'Ventral A-class motor - backward locomotion, ventral muscles.'},
    'DB1': {'type': 'motor', 'desc': 'Dorsal B-class motor - forward locomotion, dorsal muscles.'},
    'VB1': {'type': 'motor', 'desc': 'Ventral B-class motor - forward locomotion, ventral muscles.'},
    'DD1': {'type': 'motor', 'desc': 'Dorsal D-class motor - GABA inhibitory, coordinates dorsal/ventral.'},
    'VD1': {'type': 'motor', 'desc': 'Ventral D-class motor - GABA inhibitory, coordinates dorsal/ventral.'},
    'DVA': {'type': 'motor', 'desc': 'Stretch receptor and motor - proprioception, posture control.'},
    'DVB': {'type': 'motor', 'desc': 'Defecation motor neuron - controls expulsion muscles.'},
    'HSNL': {'type': 'motor', 'desc': 'Egg-laying motor neuron - serotonergic, controls vulval muscles.'},
    'HSNR': {'type': 'motor', 'desc': 'Egg-laying motor neuron - serotonergic, controls vulval muscles.'},
}

# Type classifications (fallback)
SENSORY_NEURONS = {
    'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR',
    'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASHL', 'ASHR',
    'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR',
    'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR',
    'PHAL', 'PHAR', 'PHBL', 'PHBR', 'PHCL', 'PHCR',
    'ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM',
    'ADEL', 'ADER', 'PDEL', 'PDER',
    'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR',
    'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL', 'IL1VR',
    'IL2DL', 'IL2DR', 'IL2L', 'IL2R', 'IL2VL', 'IL2VR',
    'OLLL', 'OLLR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR',
    'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR',
    'BAGL', 'BAGR', 'AQR', 'PQR',
    'FLPL', 'FLPR', 'PVDL', 'PVDR',
    'ALNL', 'ALNR', 'PLNL', 'PLNR',
}

MOTOR_NEURONS = {
    'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9',
    'VA1', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11', 'VA12',
    'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7',
    'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10', 'VB11',
    'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6',
    'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13',
    'AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11',
    'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6',
    'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMDDL', 'RMDDR',
    'RMEL', 'RMER', 'RMED', 'RMEV', 'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL', 'RMHR',
    'RIVL', 'RIVR', 'RIML', 'RIMR',
    'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR',
    'DVB', 'DVA', 'PDA', 'PDB', 'PVT', 'RID', 'AVL', 'HSNL', 'HSNR',
}

COMMAND_INTERNEURONS = {
    'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'PVCL', 'PVCR',
}


def get_neuron_type(name):
    name = name.upper().strip()
    if name in NEURON_INFO:
        return NEURON_INFO[name]['type']
    if name in SENSORY_NEURONS:
        return 'sensory'
    if name in MOTOR_NEURONS:
        return 'motor'
    if name in COMMAND_INTERNEURONS:
        return 'command'
    return 'interneuron'


def get_neuron_desc(name):
    name = name.upper().strip()
    if name in NEURON_INFO:
        return NEURON_INFO[name]['desc']
    
    ntype = get_neuron_type(name)
    if ntype == 'sensory':
        return f"Sensory neuron - detects environmental stimuli."
    elif ntype == 'motor':
        return f"Motor neuron - controls muscle contraction."
    elif ntype == 'command':
        return f"Command interneuron - controls locomotion direction."
    else:
        return f"Interneuron - processes and relays signals."


def load_connectome(data_dir):
    """Load connectome from c302 data."""
    edgelist_path = data_dir / 'herm_full_edgelist.csv'
    df = pd.read_csv(edgelist_path)
    df.columns = [c.strip() for c in df.columns]
    df['Source'] = df['Source'].str.strip().str.upper()
    df['Target'] = df['Target'].str.strip().str.upper()
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        if not source[0].isupper() or not target[0].isupper():
            continue
        weight = row.get('Weight', 1)
        conn_type = row.get('Type', 'chemical')
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight, type=conn_type)
    
    return G


def get_subgraph(G, sources, max_depth=3):
    """Get subgraph reachable from sources."""
    visited = set(sources)
    distances = {s: 0 for s in sources if s in G}
    frontier = [s for s in sources if s in G]
    
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node in frontier:
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
    
    return G.subgraph(visited).copy(), distances


def create_interactive_graph(G, sources=None, max_depth=3, output='connectome.html'):
    """Create interactive HTML visualization."""
    
    if sources:
        subgraph, distances = get_subgraph(G, sources, max_depth)
        title = f"Signal Flow from {', '.join(sources)}"
    else:
        subgraph = G
        distances = {n: 0 for n in G.nodes()}
        title = "C. elegans Connectome"
    
    # Pre-compute HIERARCHICAL layout - sources at top, then layers by hop distance
    from collections import defaultdict
    
    # Group nodes by distance
    layers = defaultdict(list)
    for node in subgraph.nodes():
        dist = distances.get(node, 0)
        layers[dist].append(node)
    
    # Position nodes in horizontal layers
    pos = {}
    y_spacing = 250  # Vertical space between layers (increased)
    
    for dist in sorted(layers.keys()):
        nodes_in_layer = sorted(layers[dist])  # Sort for consistency
        n = len(nodes_in_layer)
        x_spacing = max(120, 4000 / (n + 1))  # Much more horizontal spread
        
        for i, node in enumerate(nodes_in_layer):
            x = (i - n/2) * x_spacing
            y = dist * y_spacing
            pos[node] = (x, y)
    
    # Create pyvis network with inline CDN (works offline)
    net = Network(height='800px', width='100%', bgcolor='#1a1a2e', font_color='white',
                  directed=True, select_menu=True, cdn_resources='in_line')
    
    # Color scheme
    type_colors = {
        'sensory': '#2ecc71',      # Green
        'motor': '#e74c3c',        # Red
        'command': '#f39c12',      # Orange
        'interneuron': '#3498db',  # Blue
    }
    
    # Add nodes with pre-computed positions
    for node in subgraph.nodes():
        ntype = get_neuron_type(node)
        color = type_colors.get(ntype, '#3498db')
        
        # Size based on degree
        degree = subgraph.degree(node)
        size = 15 + min(degree * 2, 40)
        
        # Highlight sources
        is_source = sources and node in sources
        if is_source:
            color = '#00ff00'
            size = 50
        
        # Build tooltip
        desc = get_neuron_desc(node)
        in_deg = subgraph.in_degree(node)
        out_deg = subgraph.out_degree(node)
        dist = distances.get(node, '?')
        
        # Plain text tooltip (HTML doesn't render in vis.js tooltips)
        tooltip = f"""{node}
━━━━━━━━━━━━━━━━
Type: {ntype.upper()}

{desc}

Inputs: {in_deg} | Outputs: {out_deg}{f' | {dist} hops from source' if sources else ''}"""
        
        x, y = pos.get(node, (0, 0))
        
        # Show labels on sources and highly connected nodes
        label = node if (is_source or degree > 20) else ''
        
        net.add_node(node, label=label, title=tooltip, color=color, size=size,
                     group=ntype, x=x, y=y, physics=False)
    
    # Add edges
    for u, v, data in subgraph.edges(data=True):
        weight = data.get('weight', 1)
        conn_type = data.get('type', 'chemical')
        
        # Edge color based on type
        if conn_type == 'chemical':
            edge_color = '#666666'
        else:  # gap junction
            edge_color = '#9b59b6'
        
        # Width based on weight
        width = 0.5 + min(weight / 5, 3)
        
        # Highlight edges from source
        if sources and u in sources:
            edge_color = '#00ff00'
            width = 2
        
        edge_title = f"{u} -> {v} ({conn_type}, weight: {weight})"
        
        net.add_edge(u, v, title=edge_title, color=edge_color, width=width,
                     arrows='to', smooth={'type': 'curvedCW', 'roundness': 0.1})
    
    # Disable physics - we use pre-computed layout
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 12, "face": "arial", "strokeWidth": 2, "strokeColor": "#000"},
        "borderWidth": 2
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "curvedCW", "roundness": 0.1},
        "arrows": {"to": {"scaleFactor": 0.5}}
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 50,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    
    # Save
    net.write_html(output)
    print(f"\n✅ Created interactive visualization: {output}")
    print(f"   Neurons: {subgraph.number_of_nodes()}")
    print(f"   Connections: {subgraph.number_of_edges()}")
    print(f"\n   Open in browser to explore!")
    
    return net


def main():
    parser = argparse.ArgumentParser(description='Interactive C. elegans connectome explorer')
    parser.add_argument('neurons', nargs='*', help='Source neuron(s) to start from')
    parser.add_argument('--depth', type=int, default=3, help='Max hops from source (default: 3)')
    parser.add_argument('--output', '-o', type=str, default='connectome_explorer.html')
    parser.add_argument('--full', action='store_true', help='Show full connectome (no filtering)')
    parser.add_argument('--data-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Find data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / 'c302' / 'data'
    
    print(f"Loading connectome from {data_dir}...")
    G = load_connectome(data_dir)
    print(f"Loaded {G.number_of_nodes()} neurons, {G.number_of_edges()} connections")
    
    sources = None
    if args.neurons and not args.full:
        sources = [n.upper() for n in args.neurons]
        print(f"Filtering to neurons reachable from: {sources}")
    
    create_interactive_graph(G, sources=sources, max_depth=args.depth, output=args.output)


if __name__ == '__main__':
    main()
