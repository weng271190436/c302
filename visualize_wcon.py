#!/usr/bin/env python
"""
Visualize worm movement from WCON file.

Usage:
    python visualize_wcon.py path/to/worm_motion_log.wcon
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_wcon(filepath):
    """Load WCON file and extract worm positions over time."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # WCON format has 'data' array with time series
    frames = data.get('data', [])
    
    all_x = []
    all_y = []
    times = []
    
    for frame in frames:
        t = frame.get('t', 0)
        x = frame.get('x', [[]])[0]  # First worm
        y = frame.get('y', [[]])[0]
        
        if x and y:
            all_x.append(x)
            all_y.append(y)
            times.append(t)
    
    return times, all_x, all_y

def plot_trajectory(times, all_x, all_y):
    """Plot the worm's center of mass trajectory."""
    centers_x = [np.mean(x) for x in all_x]
    centers_y = [np.mean(y) for y in all_y]
    
    plt.figure(figsize=(10, 8))
    plt.plot(centers_x, centers_y, 'b-', alpha=0.5, linewidth=1)
    plt.scatter(centers_x[0], centers_y[0], c='green', s=100, label='Start', zorder=5)
    plt.scatter(centers_x[-1], centers_y[-1], c='red', s=100, label='End', zorder=5)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Worm Center of Mass Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('worm_trajectory.png', dpi=150)
    plt.show()
    print("Saved: worm_trajectory.png")

def animate_worm(times, all_x, all_y, save=True):
    """Create animation of worm movement."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find bounds
    all_x_flat = [x for frame in all_x for x in frame]
    all_y_flat = [y for frame in all_y for y in frame]
    margin = 0.1 * max(max(all_x_flat) - min(all_x_flat), max(all_y_flat) - min(all_y_flat))
    
    ax.set_xlim(min(all_x_flat) - margin, max(all_x_flat) + margin)
    ax.set_ylim(min(all_y_flat) - margin, max(all_y_flat) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Worm body line
    line, = ax.plot([], [], 'o-', color='#c4a882', linewidth=8, markersize=10, 
                    markerfacecolor='#d4b896', markeredgecolor='#a08060')
    head, = ax.plot([], [], 'o', color='#e0c8a8', markersize=15)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    title = ax.set_title('C. elegans Movement Simulation', fontsize=14)
    
    def init():
        line.set_data([], [])
        head.set_data([], [])
        time_text.set_text('')
        return line, head, time_text

    def animate(i):
        x = all_x[i]
        y = all_y[i]
        line.set_data(x, y)
        head.set_data([x[0]], [y[0]])
        time_text.set_text(f't = {times[i]:.3f}s')
        return line, head, time_text

    # Use subset of frames if too many
    skip = max(1, len(times) // 200)
    frames_to_use = range(0, len(times), skip)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames_to_use, interval=50, blit=True)
    
    if save:
        print("Saving animation (this may take a minute)...")
        anim.save('worm_movement.gif', writer='pillow', fps=20)
        print("Saved: worm_movement.gif")
    
    plt.show()
    return anim

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_wcon.py <wcon_file>")
        print("Example: python visualize_wcon.py output/C2_FW_.../worm_motion_log.wcon")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading: {filepath}")
    
    try:
        times, all_x, all_y = load_wcon(filepath)
        print(f"Loaded {len(times)} frames")
        
        if len(times) == 0:
            print("No movement data found in WCON file!")
            sys.exit(1)
        
        # Plot trajectory
        plot_trajectory(times, all_x, all_y)
        
        # Animate
        print("\nCreating animation...")
        animate_worm(times, all_x, all_y)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing WCON file: {e}")
        print("File may not be valid JSON")
    except Exception as e:
        print(f"Error: {e}")
        raise
