#!/usr/bin/env python
"""
Visualize Sibernetic/OpenWorm simulation output.

Usage:
    python visualize_sibernetic.py <output_folder>
    
Example:
    python visualize_sibernetic.py ~/repos/OpenWorm/output/C2_FW_2026-04-06_01-25-00/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def load_position_buffer(filepath):
    """Load position buffer - particle positions over time."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First ~10 lines are header/config
    # Find where actual data starts (4 floats per line)
    data_start = 0
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                [float(p) for p in parts]
                data_start = i
                break
            except:
                pass
    
    # Parse all position data
    positions = []
    current_frame = []
    
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) == 4:
            x, y, z, ptype = map(float, parts)
            current_frame.append([x, y, z])
        elif len(parts) == 1 and current_frame:
            # Separator between frames
            positions.append(np.array(current_frame))
            current_frame = []
    
    if current_frame:
        positions.append(np.array(current_frame))
    
    return positions

def load_muscle_activity(filepath):
    """Load muscle activity buffer."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    values = [float(p) for p in parts]
                    data.append(values)
                except:
                    pass
    return np.array(data)

def plot_muscle_activity(muscle_data, save_path=None):
    """Plot muscle activity heatmap over time."""
    if len(muscle_data) == 0:
        print("No muscle data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Transpose so muscles are on y-axis, time on x-axis
    im = ax.imshow(muscle_data.T, aspect='auto', cmap='hot', 
                   interpolation='nearest')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Muscle segment')
    ax.set_title('Muscle Activity Over Time')
    plt.colorbar(im, ax=ax, label='Activity')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_worm_trajectory_3d(positions, save_path=None):
    """Plot 3D trajectory of worm center of mass."""
    if len(positions) == 0:
        print("No position data")
        return
    
    centers = np.array([np.mean(frame, axis=0) for frame in positions])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 'b-', alpha=0.6, linewidth=1)
    ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], c='green', s=100, label='Start')
    ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], c='red', s=100, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Worm Center of Mass Trajectory (3D)')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_worm_snapshots(positions, n_snapshots=6, save_path=None):
    """Plot snapshots of worm body at different times."""
    if len(positions) < n_snapshots:
        n_snapshots = len(positions)
    
    indices = np.linspace(0, len(positions)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        frame = positions[idx]
        
        # Plot particles
        ax.scatter(frame[:, 0], frame[:, 1], c=frame[:, 2], 
                   cmap='viridis', s=1, alpha=0.6)
        ax.set_title(f'Frame {idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    plt.suptitle('Worm Body Snapshots Over Time', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def animate_worm_2d(positions, save_path=None, max_frames=200):
    """Create 2D animation of worm movement."""
    if len(positions) == 0:
        print("No positions to animate")
        return
    
    # Subsample if too many frames
    step = max(1, len(positions) // max_frames)
    frames_to_use = positions[::step]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find global bounds
    all_x = np.concatenate([f[:, 0] for f in frames_to_use])
    all_y = np.concatenate([f[:, 1] for f in frames_to_use])
    margin = 0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('C. elegans Movement Simulation')
    
    scatter = ax.scatter([], [], c=[], cmap='coolwarm', s=2, alpha=0.7)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text
    
    def animate(i):
        frame = frames_to_use[i]
        scatter.set_offsets(frame[:, :2])
        scatter.set_array(frame[:, 2])  # Color by Z
        time_text.set_text(f'Frame: {i * step}')
        return scatter, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames_to_use), 
                                   interval=50, blit=True)
    
    if save_path:
        print("Saving animation (this may take a minute)...")
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Saved: {save_path}")
    
    plt.show()
    return anim

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_sibernetic.py <output_folder>")
        print("Example: python visualize_sibernetic.py ~/repos/OpenWorm/output/C2_FW_.../")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    print(f"Loading from: {folder}")
    
    # Check what files exist
    position_file = folder / 'position_buffer.txt'
    muscle_file = folder / 'muscles_activity_buffer.txt'
    
    # Load muscle activity
    if muscle_file.exists():
        print("\n📊 Loading muscle activity...")
        muscle_data = load_muscle_activity(muscle_file)
        print(f"   Loaded {len(muscle_data)} timesteps, {muscle_data.shape[1] if len(muscle_data) > 0 else 0} muscles")
        
        if len(muscle_data) > 0:
            plot_muscle_activity(muscle_data, save_path='muscle_activity_heatmap.png')
    else:
        print(f"No muscle file at {muscle_file}")
    
    # Load positions
    if position_file.exists():
        print("\n🐛 Loading position data...")
        positions = load_position_buffer(position_file)
        print(f"   Loaded {len(positions)} frames")
        
        if len(positions) > 0:
            print(f"   Particles per frame: {len(positions[0])}")
            
            # Plot snapshots
            plot_worm_snapshots(positions, save_path='worm_snapshots.png')
            
            # 3D trajectory
            plot_worm_trajectory_3d(positions, save_path='worm_trajectory_3d.png')
            
            # Animation
            print("\n🎬 Creating animation...")
            animate_worm_2d(positions, save_path='worm_animation.gif')
    else:
        print(f"No position file at {position_file}")
    
    print("\n✅ Done!")

if __name__ == '__main__':
    main()
