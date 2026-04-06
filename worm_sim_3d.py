#!/usr/bin/env python
"""
3D C. elegans worm simulator with VPython.

Click neurons to stimulate them and watch the signal propagate.
Rotate view by dragging, zoom with scroll.

Usage:
    pip install vpython
    python worm_sim_3d.py
"""

from vpython import *
import numpy as np

# ============================================================================
# WORM ANATOMY DATA (simplified)
# ============================================================================

NEURONS = {
    # Sensory neurons
    'PLML': (0.95, 's'), 'PLMR': (0.95, 's'),
    'ALML': (0.15, 's'), 'ALMR': (0.15, 's'),
    'AVM': (0.12, 's'), 'PVM': (0.6, 's'),
    
    # Interneurons
    'AVAL': (0.08, 'i'), 'AVAR': (0.08, 'i'),
    'AVBL': (0.08, 'i'), 'AVBR': (0.08, 'i'),
    'PVCL': (0.92, 'i'), 'PVCR': (0.92, 'i'),
    
    # Motor neurons (subset)
    'DA1': (0.1, 'm'), 'DA3': (0.25, 'm'), 'DA5': (0.5, 'm'), 'DA7': (0.75, 'm'),
    'DB1': (0.08, 'm'), 'DB3': (0.2, 'm'), 'DB5': (0.45, 'm'), 'DB7': (0.75, 'm'),
    'VD1': (0.1, 'm'), 'VD3': (0.2, 'm'), 'VD5': (0.4, 'm'), 'VD7': (0.6, 'm'),
}

CONNECTIONS = {
    'PLML': ['PVCL', 'AVBL'],
    'PLMR': ['PVCR', 'AVBR'],
    'ALML': ['AVAL'],
    'ALMR': ['AVAR'],
    'AVM': ['AVAL', 'AVAR', 'AVBL', 'AVBR'],
    'PVM': ['AVAL', 'PVCL'],
    'AVAL': ['DA1', 'DA3', 'DA5', 'DA7', 'VD1', 'VD3', 'VD5'],
    'AVAR': ['DA1', 'DA3', 'DA5', 'DA7', 'VD1', 'VD3', 'VD5'],
    'AVBL': ['DB1', 'DB3', 'DB5', 'DB7'],
    'AVBR': ['DB1', 'DB3', 'DB5', 'DB7'],
    'PVCL': ['AVBL', 'DB5', 'DB7', 'VD5', 'VD7'],
    'PVCR': ['AVBR', 'DB5', 'DB7', 'VD5', 'VD7'],
    'DA1': ['MD01'], 'DA3': ['MD05'], 'DA5': ['MD09'], 'DA7': ['MD13'],
    'DB1': ['MD01'], 'DB3': ['MD05'], 'DB5': ['MD09'], 'DB7': ['MD13'],
    'VD1': ['MV01'], 'VD3': ['MV05'], 'VD5': ['MV09'], 'VD7': ['MV13'],
}

# Colors
COLORS = {
    's': color.orange,
    'i': color.cyan,
    'm': color.green,
}

# ============================================================================
# 3D SCENE SETUP
# ============================================================================

scene = canvas(title='C. elegans 3D Neural Simulator - Click neurons!',
               width=1200, height=700,
               center=vector(5, 0, 0),
               background=color.gray(0.9))

scene.camera.pos = vector(5, 5, 10)
scene.camera.axis = vector(0, -0.3, -1)

# Instructions
label(pos=vector(5, 4, 0), text='Click neurons (spheres) to stimulate\nDrag to rotate, scroll to zoom',
      height=12, border=4, font='sans', box=False, opacity=0)

# ============================================================================
# WORM BODY
# ============================================================================

worm_length = 10
num_segments = 20
segment_length = worm_length / num_segments

# Create body segments (cylinders)
body_segments = []
body_joints = []  # Store joint positions for bending

# Initial straight worm along X axis
for i in range(num_segments):
    x = i * segment_length
    seg = cylinder(pos=vector(x, 0, 0), 
                   axis=vector(segment_length, 0, 0),
                   radius=0.3 - i * 0.008,  # Taper toward tail
                   color=vector(0.8, 0.7, 0.6),
                   opacity=0.7)
    body_segments.append(seg)

# Head
head = sphere(pos=vector(-0.2, 0, 0), radius=0.4, color=vector(0.9, 0.8, 0.7))

# Tail
tail = cone(pos=vector(worm_length, 0, 0), axis=vector(0.5, 0, 0), 
            radius=0.15, color=vector(0.8, 0.7, 0.6))

# ============================================================================
# NEURONS (3D spheres)
# ============================================================================

neuron_objects = {}
neuron_labels = {}
neuron_activity = {n: 0.0 for n in NEURONS}

for name, (pos_along, ntype) in NEURONS.items():
    x = pos_along * worm_length
    
    # Offset based on type and L/R
    if ntype == 's':
        y = 0.8
        z = 0.3 if name.endswith('L') else -0.3 if name.endswith('R') else 0
    elif ntype == 'i':
        y = 0.5
        z = 0.4 if name.endswith('L') else -0.4 if name.endswith('R') else 0
    else:  # motor
        y = -0.5
        if name.startswith('DA'):
            z = 0.3
        elif name.startswith('DB'):
            z = 0
        else:  # VD
            z = -0.3
    
    s = sphere(pos=vector(x, y, z), radius=0.15, color=COLORS[ntype], emissive=True)
    s.name = name
    s.ntype = ntype
    s.base_color = COLORS[ntype]
    neuron_objects[name] = s
    
    # Label
    lbl = label(pos=s.pos + vector(0, 0.25, 0), text=name, height=10,
                box=False, opacity=0, color=color.black)
    neuron_labels[name] = lbl

# ============================================================================
# MUSCLES (small boxes along body)
# ============================================================================

muscle_activity = {}
muscle_objects = {}

for i in range(1, 17, 4):  # Subset of muscles
    x = (i / 20) * worm_length
    
    # Dorsal muscle (top)
    md = box(pos=vector(x, 0.5, 0), size=vector(0.3, 0.1, 0.2),
             color=color.gray(0.6), opacity=0.8)
    md_name = f'MD{i:02d}'
    muscle_objects[md_name] = md
    muscle_activity[md_name] = 0.0
    
    # Ventral muscle (bottom)
    mv = box(pos=vector(x, -0.5, 0), size=vector(0.3, 0.1, 0.2),
             color=color.gray(0.6), opacity=0.8)
    mv_name = f'MV{i:02d}'
    muscle_objects[mv_name] = mv
    muscle_activity[mv_name] = 0.0

# ============================================================================
# SIMULATION
# ============================================================================

selected = None
dt = 0.02
decay_rate = 1.5
propagation_strength = 1.5

# Segment angles for bending
segment_angles = [0.0] * num_segments

def click_handler(evt):
    """Handle mouse clicks on neurons."""
    global selected
    obj = scene.mouse.pick
    if obj and hasattr(obj, 'name'):
        neuron_activity[obj.name] = 1.0
        selected = obj.name
        print(f"Stimulated: {obj.name}")

scene.bind('click', click_handler)

# Info text
info = label(pos=vector(0, -2, 0), text='', height=12, box=False)

# Main loop
print("3D Worm Simulator Running!")
print("Click on neurons to stimulate them.")
print("Drag to rotate view, scroll to zoom.")

while True:
    rate(60)
    
    # Propagate neural activity
    new_activity = {n: neuron_activity[n] for n in NEURONS}
    new_muscle = {m: muscle_activity[m] for m in muscle_activity}
    
    for source, targets in CONNECTIONS.items():
        if source in neuron_activity and neuron_activity[source] > 0.1:
            for target in targets:
                if target in new_activity:
                    new_activity[target] = min(1.0, new_activity[target] + 
                        neuron_activity[source] * propagation_strength * dt * 3)
                elif target in new_muscle:
                    new_muscle[target] = min(1.0, new_muscle[target] + 
                        neuron_activity[source] * propagation_strength * dt * 3)
    
    # Decay
    for n in NEURONS:
        new_activity[n] = max(0, new_activity[n] - decay_rate * dt)
    for m in muscle_activity:
        new_muscle[m] = max(0, new_muscle[m] - decay_rate * dt)
    
    neuron_activity = new_activity
    muscle_activity = new_muscle
    
    # Update neuron visuals
    for name, obj in neuron_objects.items():
        activity = neuron_activity[name]
        if activity > 0.1:
            obj.color = vector(1, 0.2, 0.2)  # Red when active
            obj.radius = 0.15 + activity * 0.08
        else:
            obj.color = obj.base_color
            obj.radius = 0.15
    
    # Update muscle visuals
    for name, obj in muscle_objects.items():
        activity = muscle_activity.get(name, 0)
        r = 0.5 + activity * 0.5
        g = 0.5 - activity * 0.4
        b = 0.5 - activity * 0.4
        obj.color = vector(r, g, b)
    
    # Update body bending based on muscles
    for i in range(num_segments):
        seg_idx = int((i / num_segments) * 16) + 1
        md_name = f'MD{seg_idx:02d}'
        mv_name = f'MV{seg_idx:02d}'
        
        dorsal = muscle_activity.get(md_name, 0)
        ventral = muscle_activity.get(mv_name, 0)
        
        # Target angle (ventral - dorsal because dorsal contracts = bend up)
        target_angle = (ventral - dorsal) * 0.3
        segment_angles[i] += (target_angle - segment_angles[i]) * 0.1
    
    # Rebuild body shape
    pos = vector(0, 0, 0)
    angle = 0
    
    for i, seg in enumerate(body_segments):
        angle += segment_angles[i]
        direction = vector(cos(angle), sin(angle), 0) * segment_length
        seg.pos = pos
        seg.axis = direction
        pos = pos + direction
    
    # Update head and tail
    head.pos = body_segments[0].pos - vector(0.2, 0, 0)
    tail.pos = body_segments[-1].pos + body_segments[-1].axis
    tail.axis = body_segments[-1].axis.norm() * 0.5
    
    # Update info
    if selected:
        info.text = f"Selected: {selected} | Activity: {neuron_activity.get(selected, 0):.2f}"
