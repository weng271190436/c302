#!/usr/bin/env python
"""
Interactive C. elegans worm simulator with pygame.

Click neurons to stimulate them and watch the signal propagate through the
connectome. Muscles contract and the worm body bends!

Usage:
    pip install pygame numpy
    python worm_sim.py

Controls:
    - Click on neurons (colored dots) to stimulate
    - Press R to reset
    - Press Q or ESC to quit
    - Press SPACE to pause/resume
"""

import math
import sys
import numpy as np

try:
    import pygame
except ImportError:
    print("pygame not installed. Run: pip install pygame")
    sys.exit(1)

# ============================================================================
# WORM ANATOMY DATA
# ============================================================================

# Key neurons and their approximate positions along the body (0=head, 1=tail)
# Format: name -> (position_along_body, layer: 's'ensory/'i'nterneuron/'m'otor)
NEURONS = {
    # Sensory neurons (head)
    'PLML': (0.95, 's'), 'PLMR': (0.95, 's'),  # Tail touch
    'ALML': (0.15, 's'), 'ALMR': (0.15, 's'),  # Anterior touch
    'AVM': (0.12, 's'),                         # Anterior touch
    'PVM': (0.6, 's'),                          # Posterior touch
    
    # Interneurons (command neurons)
    'AVAL': (0.08, 'i'), 'AVAR': (0.08, 'i'),  # Backward command
    'AVBL': (0.08, 'i'), 'AVBR': (0.08, 'i'),  # Forward command
    'AVDL': (0.08, 'i'), 'AVDR': (0.08, 'i'),  # Backward command
    'PVCL': (0.92, 'i'), 'PVCR': (0.92, 'i'),  # Tail interneurons
    
    # Motor neurons (along body)
    'DA1': (0.1, 'm'), 'DA2': (0.15, 'm'), 'DA3': (0.25, 'm'),
    'DA4': (0.35, 'm'), 'DA5': (0.5, 'm'), 'DA6': (0.65, 'm'),
    'DA7': (0.75, 'm'), 'DA8': (0.85, 'm'), 'DA9': (0.9, 'm'),
    'DB1': (0.08, 'm'), 'DB2': (0.12, 'm'), 'DB3': (0.2, 'm'),
    'DB4': (0.3, 'm'), 'DB5': (0.45, 'm'), 'DB6': (0.6, 'm'), 'DB7': (0.75, 'm'),
    'VD1': (0.1, 'm'), 'VD2': (0.15, 'm'), 'VD3': (0.2, 'm'),
    'VD4': (0.3, 'm'), 'VD5': (0.4, 'm'), 'VD6': (0.5, 'm'),
    'VD7': (0.6, 'm'), 'VD8': (0.7, 'm'), 'VD9': (0.75, 'm'),
    'VD10': (0.8, 'm'), 'VD11': (0.85, 'm'), 'VD12': (0.9, 'm'),
}

# Simplified connectome (which neurons excite which)
# In reality there are thousands of connections - this is a subset for visualization
CONNECTIONS = {
    # Touch response circuit - SYMMETRIC pathways, different muscles
    'PLML': ['PVCL', 'AVBL'],  # Left touch → left circuit
    'PLMR': ['PVCR', 'AVBR'],  # Right touch → right circuit
    'ALML': ['AVAL', 'AVDL'],
    'ALMR': ['AVAR', 'AVDR'],
    'AVM': ['AVAL', 'AVAR', 'AVBL', 'AVBR'],
    'PVM': ['AVAL', 'AVAR', 'PVCL'],
    
    # Command interneurons - EQUAL number of motor neuron targets
    'AVAL': ['DA1', 'DA3', 'DA5', 'DA7', 'DA9'],
    'AVAR': ['DA2', 'DA4', 'DA6', 'DA8'],
    'AVBL': ['DB1', 'DB3', 'DB5', 'DB7'],
    'AVBR': ['DB2', 'DB4', 'DB6'],
    'AVDL': ['DA1', 'DA3'],
    'AVDR': ['DA2', 'DA4'],
    'PVCL': ['AVBL', 'DB1', 'DB3', 'DB5', 'DB7'],  # Direct to motors too
    'PVCR': ['AVBR', 'DB2', 'DB4', 'DB6'],          # Direct to motors too
    
    # Motor neurons to muscles - LEFT side (4 muscles each)
    'DB1': ['MDL01', 'MDL02', 'MDL03', 'MDL04'],
    'DB3': ['MDL05', 'MDL06', 'MDL07', 'MDL08'],
    'DB5': ['MDL09', 'MDL10', 'MDL11', 'MDL12'],
    'DB7': ['MDL13', 'MDL14', 'MDL15', 'MDL16'],
    
    # Motor neurons to muscles - RIGHT side (4 muscles each)
    'DB2': ['MDR01', 'MDR02', 'MDR03', 'MDR04'],
    'DB4': ['MDR05', 'MDR06', 'MDR07', 'MDR08'],
    'DB6': ['MDR09', 'MDR10', 'MDR11', 'MDR12'],
    
    # DA neurons for backward motion
    'DA1': ['MDL01', 'MDL02'], 'DA3': ['MDL05', 'MDL06'],
    'DA5': ['MDL09', 'MDL10'], 'DA7': ['MDL13', 'MDL14'],
    'DA9': ['MDL17', 'MDL18'],
    'DA2': ['MDR01', 'MDR02'], 'DA4': ['MDR05', 'MDR06'],
    'DA6': ['MDR09', 'MDR10'], 'DA8': ['MDR13', 'MDR14'],
    
    # VD neurons
    'VD1': ['MVL01', 'MVR01'], 'VD2': ['MVL02', 'MVR02'],
    'VD3': ['MVL03', 'MVR03'], 'VD4': ['MVL04', 'MVR04'],
    'VD5': ['MVL06', 'MVR06'], 'VD6': ['MVL08', 'MVR08'],
    'VD7': ['MVL10', 'MVR10'], 'VD8': ['MVL12', 'MVR12'],
    'VD9': ['MVL14', 'MVR14'], 'VD10': ['MVL16', 'MVR16'],
    'VD11': ['MVL18', 'MVR18'], 'VD12': ['MVL20', 'MVR20'],
}

# Muscles along the body (24 segments, dorsal left/right, ventral left/right)
MUSCLES = []
for i in range(1, 21):
    MUSCLES.extend([f'MDL{i:02d}', f'MDR{i:02d}', f'MVL{i:02d}', f'MVR{i:02d}'])

# ============================================================================
# COLORS
# ============================================================================

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
DARK_GRAY = (40, 40, 40)
LIGHT_BG = (245, 243, 240)  # Cream background
TEXT_COLOR = (50, 50, 50)   # Dark text

# Neuron colors by type
NEURON_COLORS = {
    's': (255, 200, 100),   # Sensory: orange
    'i': (100, 200, 255),   # Interneuron: blue
    'm': (150, 255, 150),   # Motor: green
}

ACTIVE_COLOR = (255, 50, 50)  # Red when firing
MUSCLE_COLOR = (200, 100, 100)  # Muscle default
MUSCLE_ACTIVE = (255, 50, 50)   # Muscle contracting
WORM_COLOR = (180, 150, 120)    # Worm body

# ============================================================================
# SIMULATION STATE
# ============================================================================

class WormSimulator:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("C. elegans Neural Simulator - Click neurons to stimulate!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 12)
        self.big_font = pygame.font.SysFont('Arial', 18)
        
        # Simulation state
        self.neuron_activity = {n: 0.0 for n in NEURONS}  # 0-1 activation
        self.muscle_activity = {m: 0.0 for m in MUSCLES}  # 0-1 activation
        self.paused = False
        self.selected_neuron = None
        
        # Worm body segments
        self.num_segments = 20
        self.segment_angles = [0.0] * self.num_segments
        
        # Pre-compute neuron screen positions
        self.neuron_positions = {}
        self._compute_neuron_positions()
        
    def _compute_neuron_positions(self):
        """Compute screen positions for neurons based on worm body."""
        margin = 320  # More margin = smaller worm, more room to move
        worm_length = self.width - 2 * margin
        worm_y = self.height // 2 + 40
        
        # Manual positioning for cleaner layout
        # Head interneurons - spread them out in a grid
        head_interneurons = ['AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR']
        head_x_start = margin - 30
        
        for i, name in enumerate(head_interneurons):
            row = i // 2  # 0, 0, 1, 1, 2, 2
            col = i % 2   # 0, 1, 0, 1, 0, 1
            x = head_x_start + col * 40
            y = worm_y - 80 + row * 30
            self.neuron_positions[name] = (int(x), int(y))
        
        # Tail interneurons
        self.neuron_positions['PVCL'] = (int(margin + 0.92 * worm_length - 20), worm_y - 60)
        self.neuron_positions['PVCR'] = (int(margin + 0.92 * worm_length + 20), worm_y - 60)
        
        # Sensory neurons - spread out
        self.neuron_positions['PLML'] = (int(margin + 0.95 * worm_length), worm_y - 100)
        self.neuron_positions['PLMR'] = (int(margin + 0.95 * worm_length + 40), worm_y - 80)
        self.neuron_positions['ALML'] = (int(margin + 0.12 * worm_length), worm_y - 120)
        self.neuron_positions['ALMR'] = (int(margin + 0.18 * worm_length), worm_y - 120)
        self.neuron_positions['AVM'] = (int(margin + 0.10 * worm_length), worm_y - 100)
        self.neuron_positions['PVM'] = (int(margin + 0.60 * worm_length), worm_y - 80)
        
        # Motor neurons - evenly spaced along bottom
        motor_neurons = [
            ('DA1', 0.10), ('DA2', 0.15), ('DA3', 0.25), ('DA4', 0.35),
            ('DA5', 0.50), ('DA6', 0.65), ('DA7', 0.75), ('DA8', 0.85), ('DA9', 0.90),
            ('DB1', 0.08), ('DB2', 0.12), ('DB3', 0.20), ('DB4', 0.30),
            ('DB5', 0.45), ('DB6', 0.60), ('DB7', 0.75),
            ('VD1', 0.10), ('VD2', 0.15), ('VD3', 0.20), ('VD4', 0.30),
            ('VD5', 0.40), ('VD6', 0.50), ('VD7', 0.60), ('VD8', 0.70),
            ('VD9', 0.75), ('VD10', 0.80), ('VD11', 0.85), ('VD12', 0.90),
        ]
        
        # Sort by position and assign evenly spaced x coords
        motor_neurons.sort(key=lambda x: x[1])
        num_motors = len(motor_neurons)
        motor_spacing = worm_length / (num_motors + 1)
        
        for i, (name, _) in enumerate(motor_neurons):
            x = margin + (i + 1) * motor_spacing
            # Stagger rows: DA=top, DB=middle, VD=bottom
            if name.startswith('DA'):
                y = worm_y + 70
            elif name.startswith('DB'):
                y = worm_y + 95
            else:  # VD
                y = worm_y + 120
            self.neuron_positions[name] = (int(x), int(y))
    
    def stimulate_neuron(self, name):
        """Stimulate a neuron (set to max activation)."""
        if name in self.neuron_activity:
            self.neuron_activity[name] = 1.0
            self.selected_neuron = name
    
    def update(self, dt):
        """Update simulation state."""
        if self.paused:
            return
            
        decay_rate = 1.5  # How fast activity decays (slower = longer signal)
        propagation_strength = 1.5  # How much activity propagates (stronger cascade)
        
        # Propagate activity through connections
        new_activity = {n: self.neuron_activity[n] for n in NEURONS}
        new_muscle = {m: self.muscle_activity[m] for m in MUSCLES}
        
        for source, targets in CONNECTIONS.items():
            if source in self.neuron_activity and self.neuron_activity[source] > 0.1:
                for target in targets:
                    if target in new_activity:
                        # Propagate to neurons with delay effect
                        new_activity[target] = min(1.0, new_activity[target] + 
                            self.neuron_activity[source] * propagation_strength * dt * 3)
                    elif target in new_muscle:
                        # Propagate to muscles
                        new_muscle[target] = min(1.0, new_muscle[target] + 
                            self.neuron_activity[source] * propagation_strength * dt * 3)
        
        # Apply decay
        for n in NEURONS:
            new_activity[n] = max(0, new_activity[n] - decay_rate * dt)
        for m in MUSCLES:
            new_muscle[m] = max(0, new_muscle[m] - decay_rate * dt)
        
        self.neuron_activity = new_activity
        self.muscle_activity = new_muscle
        
        # Update worm body shape based on muscle activity
        self._update_body_shape()
    
    def _update_body_shape(self):
        """Compute body bend based on muscle activation."""
        for i in range(self.num_segments):
            # Get muscle activity for this segment
            seg_idx = i + 1
            dorsal_l = self.muscle_activity.get(f'MDL{seg_idx:02d}', 0)
            dorsal_r = self.muscle_activity.get(f'MDR{seg_idx:02d}', 0)
            ventral_l = self.muscle_activity.get(f'MVL{seg_idx:02d}', 0)
            ventral_r = self.muscle_activity.get(f'MVR{seg_idx:02d}', 0)
            
            # Left vs right imbalance causes turning
            left_total = dorsal_l + ventral_l
            right_total = dorsal_r + ventral_r
            
            # Dorsal vs ventral imbalance causes dorsal/ventral bending
            dorsal = (dorsal_l + dorsal_r) / 2
            ventral = (ventral_l + ventral_r) / 2
            
            # Combined: L/R difference + D/V difference
            target_angle = (left_total - right_total) * 0.3 + (dorsal - ventral) * 0.25
            
            # Smooth transition
            self.segment_angles[i] += (target_angle - self.segment_angles[i]) * 0.1
    
    def draw(self):
        """Render the simulation."""
        self.screen.fill(LIGHT_BG)
        
        # Draw title
        title = self.big_font.render(
            "C. elegans Neural Simulator - Click neurons to stimulate!", 
            True, TEXT_COLOR)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 10))
        
        # Draw legend
        self._draw_legend()
        
        # Draw worm body
        self._draw_worm_body()
        
        # Draw muscles
        self._draw_muscles()
        
        # Draw connections (faint lines)
        self._draw_connections()
        
        # Draw neurons
        self._draw_neurons()
        
        # Draw info panel
        self._draw_info_panel()
        
        pygame.display.flip()
    
    def _draw_legend(self):
        """Draw color legend."""
        x, y = 20, 50
        items = [
            ("Sensory", NEURON_COLORS['s']),
            ("Interneuron", NEURON_COLORS['i']),
            ("Motor", NEURON_COLORS['m']),
            ("Active", ACTIVE_COLOR),
        ]
        for label, color in items:
            pygame.draw.circle(self.screen, color, (x + 8, y + 8), 8)
            text = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(text, (x + 22, y + 2))
            y += 22
        
        # Controls
        y += 10
        controls = ["R: Reset", "SPACE: Pause", "Q: Quit"]
        for ctrl in controls:
            text = self.font.render(ctrl, True, GRAY)
            self.screen.blit(text, (x, y))
            y += 18
    
    def _draw_worm_body(self):
        """Draw the worm body as connected segments."""
        margin = 320
        worm_length = self.width - 2 * margin
        segment_length = worm_length / self.num_segments
        
        # Start position
        x, y = margin, self.height // 2
        angle = 0
        
        points = [(x, y)]
        
        for i in range(self.num_segments):
            angle += self.segment_angles[i]
            x += segment_length * math.cos(angle)
            y += segment_length * math.sin(angle)
            points.append((x, y))
        
        # Draw thick worm body
        if len(points) > 1:
            # Draw circles at each point to fill gaps
            for i, (px, py) in enumerate(points):
                # Taper at tail
                radius = 12 if i < len(points) - 3 else 12 - (i - (len(points) - 3)) * 3
                radius = max(4, radius)
                pygame.draw.circle(self.screen, WORM_COLOR, (int(px), int(py)), radius)
                pygame.draw.circle(self.screen, (200, 180, 150), (int(px), int(py)), radius - 3)
        
        # Draw head
        pygame.draw.circle(self.screen, (220, 200, 170), 
                          (int(points[0][0]), int(points[0][1])), 18)
        # Draw tail
        pygame.draw.circle(self.screen, WORM_COLOR,
                          (int(points[-1][0]), int(points[-1][1])), 8)
        
        # Store points for muscle drawing
        self.body_points = points
    
    def _draw_muscles(self):
        """Draw muscles along the body."""
        if not hasattr(self, 'body_points') or len(self.body_points) < 2:
            return
        
        margin = 320
        worm_length = self.width - 2 * margin
        
        for i in range(1, 21):  # Muscles 01-20
            # Position along body (0-1)
            pos = i / 21
            
            # Find corresponding body segment
            seg_idx = int(pos * (len(self.body_points) - 1))
            if seg_idx >= len(self.body_points) - 1:
                seg_idx = len(self.body_points) - 2
            
            # Interpolate position on body
            p1 = self.body_points[seg_idx]
            p2 = self.body_points[seg_idx + 1]
            t = (pos * (len(self.body_points) - 1)) - seg_idx
            
            base_x = p1[0] + t * (p2[0] - p1[0])
            base_y = p1[1] + t * (p2[1] - p1[1])
            
            # Calculate perpendicular direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = max(1, math.sqrt(dx*dx + dy*dy))
            perp_x = -dy / length
            perp_y = dx / length
            
            # Draw 2 muscles at this position (MD = dorsal, MV = ventral)
            # Combine L/R activity for display
            muscle_offset = 22
            
            # Dorsal muscle (above) - combine MDL + MDR
            mdl_act = self.muscle_activity.get(f'MDL{i:02d}', 0)
            mdr_act = self.muscle_activity.get(f'MDR{i:02d}', 0)
            dorsal_activity = max(mdl_act, mdr_act)
            
            # Ventral muscle (below) - combine MVL + MVR
            mvl_act = self.muscle_activity.get(f'MVL{i:02d}', 0)
            mvr_act = self.muscle_activity.get(f'MVR{i:02d}', 0)
            ventral_activity = max(mvl_act, mvr_act)
            
            muscles_at_pos = [
                ('MD', 0, -muscle_offset, dorsal_activity),   # Dorsal (above)
                ('MV', 0, muscle_offset, ventral_activity),   # Ventral (below)
            ]
            
            for label, ox, oy, activity in muscles_at_pos:
                # Color based on activity
                r = int(150 + activity * 105)
                g = int(150 - activity * 100)
                b = int(150 - activity * 100)
                color = (min(255, r), max(0, g), max(0, b))
                
                mx = base_x + ox
                my = base_y + oy
                
                # Draw muscle as small rectangle
                size = 4 + int(activity * 4)
                pygame.draw.rect(self.screen, color, 
                               (int(mx - size/2), int(my - size/2), size, size))
    
    def _draw_connections(self):
        """Draw synaptic connections as faint lines."""
        for source, targets in CONNECTIONS.items():
            if source not in self.neuron_positions:
                continue
            sx, sy = self.neuron_positions[source]
            activity = self.neuron_activity.get(source, 0)
            
            for target in targets:
                if target not in self.neuron_positions:
                    continue
                tx, ty = self.neuron_positions[target]
                
                # Brighter if active
                alpha = int(30 + activity * 200)
                color = (alpha, alpha // 2, alpha // 2)
                pygame.draw.line(self.screen, color, (sx, sy), (tx, ty), 1)
    
    def _draw_neurons(self):
        """Draw all neurons."""
        for name, (pos, ntype) in NEURONS.items():
            x, y = self.neuron_positions[name]
            activity = self.neuron_activity[name]
            
            # Interpolate color based on activity
            base_color = NEURON_COLORS[ntype]
            r = int(base_color[0] + (ACTIVE_COLOR[0] - base_color[0]) * activity)
            g = int(base_color[1] + (ACTIVE_COLOR[1] - base_color[1]) * activity)
            b = int(base_color[2] + (ACTIVE_COLOR[2] - base_color[2]) * activity)
            color = (min(255, r), min(255, g), min(255, b))
            
            # Draw neuron
            radius = 8 + int(activity * 6)  # Bigger when active
            pygame.draw.circle(self.screen, color, (x, y), radius)
            pygame.draw.circle(self.screen, WHITE, (x, y), radius, 1)
            
            # Draw label
            label = self.font.render(name, True, TEXT_COLOR)
            self.screen.blit(label, (x - label.get_width()//2, y - radius - 14))
    
    def _draw_info_panel(self):
        """Draw info about selected neuron."""
        x, y = self.width - 250, 50
        
        if self.selected_neuron:
            name = self.selected_neuron
            pos, ntype = NEURONS.get(name, (0, '?'))
            
            type_names = {'s': 'Sensory', 'i': 'Interneuron', 'm': 'Motor'}
            
            lines = [
                f"Selected: {name}",
                f"Type: {type_names.get(ntype, 'Unknown')}",
                f"Position: {pos:.0%} along body",
                f"Activity: {self.neuron_activity.get(name, 0):.2f}",
                "",
                "Connects to:",
            ]
            
            targets = CONNECTIONS.get(name, [])
            for t in targets[:8]:  # Show first 8
                lines.append(f"  → {t}")
            if len(targets) > 8:
                lines.append(f"  ... and {len(targets)-8} more")
            
            for i, line in enumerate(lines):
                text = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(text, (x, y + i * 18))
        
        # Pause indicator
        if self.paused:
            text = self.big_font.render("PAUSED", True, (200, 50, 50))  # Red
            self.screen.blit(text, (self.width//2 - text.get_width()//2, 
                                   self.height - 40))
    
    def get_neuron_at_pos(self, pos):
        """Find neuron at screen position."""
        mx, my = pos
        for name, (nx, ny) in self.neuron_positions.items():
            dist = math.sqrt((mx - nx)**2 + (my - ny)**2)
            if dist < 15:
                return name
        return None
    
    def reset(self):
        """Reset all activity."""
        self.neuron_activity = {n: 0.0 for n in NEURONS}
        self.muscle_activity = {m: 0.0 for m in MUSCLES}
        self.segment_angles = [0.0] * self.num_segments
        self.selected_neuron = None
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    neuron = self.get_neuron_at_pos(event.pos)
                    if neuron:
                        self.stimulate_neuron(neuron)
            
            self.update(dt)
            self.draw()
        
        pygame.quit()


def main():
    print("Starting C. elegans Neural Simulator...")
    print("Click on neurons to stimulate them!")
    print("Watch the signal propagate through the connectome.")
    print()
    
    sim = WormSimulator()
    sim.run()


if __name__ == "__main__":
    main()
