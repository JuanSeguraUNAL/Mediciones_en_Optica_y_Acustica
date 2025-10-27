"""
Práctica 7 - Visualización de mapas de color de parámetros de elipses
Genera mapas de color para Emax, Emin, eccentricity, Phi y R² para cada canal RGB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Configuration
ROI_width = 780
ROI_height = 780

print("="*60)
print("Generating colormaps for ellipse parameters")
print("="*60)

# Load data
print("\nLoading CSV files...")
df_red = pd.read_csv("./Practica7_resultados/ellipses_red.csv")
df_green = pd.read_csv("./Practica7_resultados/ellipses_green.csv")
df_blue = pd.read_csv("./Practica7_resultados/ellipses_blue.csv")

print(f"  Red channel: {len(df_red)} pixels")
print(f"  Green channel: {len(df_green)} pixels")
print(f"  Blue channel: {len(df_blue)} pixels")

# Parameters to plot
parameters = ['Emax', 'Emin', 'eccentricity', 'Phi', 'R2']
param_labels = ['$E_{max}$', '$E_{min}$', 'Eccentricity', r'$\Phi$ (°)', '$R^2$']

# Channel data and colors
channels = [
    ('Red', df_red, 'Reds'),
    ('Green', df_green, 'Greens'),
    ('Blue', df_blue, 'Blues')
]

print("\nPre-processing data and computing global scales...")

# Pre-create all parameter maps efficiently (vectorized)
param_maps = {}
global_vmin = {}
global_vmax = {}

for param in parameters:
    param_maps[param] = {}
    all_values = []
    
    for channel_name, df, _ in channels:
        # Vectorized map creation
        param_map = np.full((ROI_height, ROI_width), np.nan, dtype=np.float32)
        
        # Convert to numpy arrays for faster indexing
        x_coords = df['x'].values.astype(int)
        y_coords = df['y'].values.astype(int)
        values = df[param].values
        
        # Vectorized assignment (much faster than iterrows)
        valid_mask = (x_coords >= 0) & (x_coords < ROI_width) & (y_coords >= 0) & (y_coords < ROI_height)
        param_map[y_coords[valid_mask], x_coords[valid_mask]] = values[valid_mask]
        
        param_maps[param][channel_name] = param_map
        all_values.append(values)
    
    # Compute global min/max for this parameter across all channels
    all_values = np.concatenate(all_values)
    global_vmin[param] = np.nanmin(all_values)
    global_vmax[param] = np.nanmax(all_values)
    
    print(f"  {param}: vmin={global_vmin[param]:.3f}, vmax={global_vmax[param]:.3f}")

# Create figure with 3x5 grid
print("\nGenerating plots...")
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for row_idx, (channel_name, df, cmap_base) in enumerate(channels):
    print(f"  Plotting {channel_name} channel...")
    
    for col_idx, (param, param_label) in enumerate(zip(parameters, param_labels)):
        ax = axes[row_idx, col_idx]
        
        # Use viridis for R2, otherwise use channel-specific colormap
        if param == 'R2':
            cmap = 'viridis'
        else:
            cmap = cmap_base
        
        # Get pre-computed map
        param_map = param_maps[param][channel_name]
        
        # Plot with shared scale
        im = ax.imshow(param_map, cmap=cmap, origin='upper', aspect='equal',
                      vmin=global_vmin[param], vmax=global_vmax[param])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        # Title and labels
        if row_idx == 0:
            ax.set_title(param_label, fontsize=14, fontweight='bold')
        
        if col_idx == 0:
            ax.set_ylabel(f'{channel_name} Channel', fontsize=12, fontweight='bold')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add text with statistics
        valid_data = param_map[~np.isnan(param_map)]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Main title - determine if test or full mode based on number of pixels
n_pixels = len(df_red)
total_pixels = ROI_width * ROI_height
if n_pixels < total_pixels:
    mode_text = f'Test Mode ({n_pixels:,} pixels)'
else:
    mode_text = f'Full ROI ({n_pixels:,} pixels)'

fig.suptitle(f'Ellipse Parameters Spatial Distribution - {mode_text}', 
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_path = "./Practica7_resultados/ellipse_parameters_colormaps.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

print("\n" + "="*60)
print("Colormap generation complete!")
print("="*60)
