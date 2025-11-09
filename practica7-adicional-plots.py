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

print("\nGenerating plots...")

for param, param_label in zip(parameters, param_labels):
    print(f"  Plotting parameter: {param}")
    
    # Crear figura con 1 fila (RGB) y 3 columnas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    
    for col_idx, (channel_name, df, cmap_base) in enumerate(channels):
        ax = axes[col_idx]
        
        # Seleccionar el mapa correspondiente
        param_map = param_maps[param][channel_name]
        
        # Seleccionar colormap
        cmap = 'viridis' if param == 'R2' else cmap_base
        
        # Graficar
        im = ax.imshow(param_map, cmap=cmap, origin='upper', aspect='equal',
                      vmin=global_vmin[param], vmax=global_vmax[param])
        
        # Añadir colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        # Quitar ticks y títulos
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")  # sin título
        
        # Añadir estadísticas
        valid_data = param_map[~np.isnan(param_map)]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Guardar figura sin título general
    output_path = f"./Practica7_resultados/{param}_colormap.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    → Saved: {output_path}")

print(f"\nPlot saved to: {output_path}")

print("\n" + "="*60)
print("Colormap generation complete!")
print("="*60)
