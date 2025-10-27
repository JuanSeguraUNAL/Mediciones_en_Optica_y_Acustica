"""
Práctica 7 - Cálculo de elipses de polarización para todos los píxeles del ROI
Procesamiento paralelo con GPU usando CuPy
"""

import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import time

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU detected! Using CuPy for acceleration")
except ImportError:
    print("CuPy not available. Using NumPy (CPU only)")
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

# ROI Configuration (same as notebook)
ROI_x = 990
ROI_y = 670
ROI_width = 780
ROI_height = 780

# Test mode: process only first N pixels
TEST_MODE = False
TEST_PIXELS = 10000

print(f"\n{'='*60}")
print(f"ROI Configuration:")
print(f"  Position: ({ROI_x}, {ROI_y})")
print(f"  Size: {ROI_width} × {ROI_height} pixels")
print(f"  Total pixels: {ROI_width * ROI_height:,}")
print(f"\nMode: {'TEST (first 1000 pixels)' if TEST_MODE else 'FULL ROI'}")
print(f"{'='*60}\n")


def load_all_images(angles):
    """Load all images and extract ROI for each angle"""
    print("Loading all images...")
    images_R = []
    images_G = []
    images_B = []
    
    for angle in tqdm(angles, desc="Loading images"):
        img_path = f"Practica7_datos/{angle}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Extract ROI
            roi = img_array[ROI_y:ROI_y+ROI_height, ROI_x:ROI_x+ROI_width]
            
            # Separate channels
            images_R.append(roi[:, :, 0])
            images_G.append(roi[:, :, 1])
            images_B.append(roi[:, :, 2])
    
    # Stack into 3D arrays: (n_angles, height, width)
    images_R = np.stack(images_R, axis=0)
    images_G = np.stack(images_G, axis=0)
    images_B = np.stack(images_B, axis=0)
    
    print(f"Loaded {len(angles)} images")
    print(f"Shape per channel: {images_R.shape}")
    
    return images_R, images_G, images_B, angles


def calculate_ellipse_params_gpu(E_values, angles):
    """
    Calculate ellipse parameters for a batch of pixels using GPU
    E_values: array of shape (n_pixels, n_angles)
    Returns: Emax, Emin, e, Phi, R2 arrays of shape (n_pixels,)
    """
    if GPU_AVAILABLE:
        E_values = cp.asarray(E_values)
        angles_array = cp.asarray(angles)
        xp = cp
    else:
        angles_array = np.array(angles)
        xp = np
    
    # Find minimum E value (minor axis) for each pixel
    min_idx = xp.argmin(E_values, axis=1)
    E_min = xp.take_along_axis(E_values, min_idx[:, None], axis=1).squeeze()
    
    # Get angle at minimum for each pixel - use advanced indexing
    n_pixels = E_values.shape[0]
    angle_min = angles_array[min_idx]
    
    # Calculate angles at ±90° from minimum
    angle_90_plus = (angle_min + 90) % 360
    angle_90_minus = (angle_min - 90) % 360
    
    # Find closest angles in dataset to ±90° positions
    diff_plus = xp.abs(angles_array[None, :] - angle_90_plus[:, None])
    diff_minus = xp.abs(angles_array[None, :] - angle_90_minus[:, None])
    idx_90_plus = xp.argmin(diff_plus, axis=1)
    idx_90_minus = xp.argmin(diff_minus, axis=1)
    
    # Get E values at ±90°
    E_90_plus = xp.take_along_axis(E_values, idx_90_plus[:, None], axis=1).squeeze()
    E_90_minus = xp.take_along_axis(E_values, idx_90_minus[:, None], axis=1).squeeze()
    
    # Choose the higher value as major axis
    use_plus = E_90_plus >= E_90_minus
    E_max = xp.where(use_plus, E_90_plus, E_90_minus)
    angle_max_idx = xp.where(use_plus, idx_90_plus, idx_90_minus)
    Phi = angles_array[angle_max_idx]
    
    # Calculate eccentricity
    e = xp.sqrt(xp.maximum(E_max**2 - E_min**2, 0)) / xp.maximum(E_max, 1e-10)
    
    # Calculate R² - goodness of fit
    # Ellipse equation: E(θ) = (a*b) / sqrt((b*cos(θ-φ))^2 + (a*sin(θ-φ))^2)
    # where a = E_max, b = E_min, φ = Phi
    angles_rad = xp.deg2rad(angles_array)
    Phi_rad = xp.deg2rad(Phi)
    
    # Calculate predicted E values for all angles
    # angles_rad: (n_angles,), Phi_rad: (n_pixels,)
    # Need broadcasting: (n_pixels, n_angles)
    theta_diff = angles_rad[None, :] - Phi_rad[:, None]  # (n_pixels, n_angles)
    
    a = E_max[:, None]  # (n_pixels, 1)
    b = E_min[:, None]  # (n_pixels, 1)
    
    denominator = xp.sqrt((b * xp.cos(theta_diff))**2 + (a * xp.sin(theta_diff))**2)
    E_predicted = (a * b) / xp.maximum(denominator, 1e-10)  # (n_pixels, n_angles)
    
    # Calculate R² using correlation coefficient squared (always between 0 and 1)
    # This measures how well the ellipse shape correlates with the data
    # R² = (correlation between E_measured and E_predicted)²
    
    # For each pixel, calculate correlation
    E_mean_measured = xp.mean(E_values, axis=1, keepdims=True)
    E_mean_predicted = xp.mean(E_predicted, axis=1, keepdims=True)
    
    numerator = xp.sum((E_values - E_mean_measured) * (E_predicted - E_mean_predicted), axis=1)
    denominator_measured = xp.sqrt(xp.sum((E_values - E_mean_measured)**2, axis=1))
    denominator_predicted = xp.sqrt(xp.sum((E_predicted - E_mean_predicted)**2, axis=1))
    
    # Pearson correlation coefficient
    r = numerator / xp.maximum(denominator_measured * denominator_predicted, 1e-10)
    
    # R² is the square of correlation coefficient (always 0 to 1)
    R2 = r**2
    
    # Clip to [0, 1] range to handle numerical errors
    R2 = xp.clip(R2, 0.0, 1.0)
    
    # Convert back to numpy if using GPU
    if GPU_AVAILABLE:
        E_max = cp.asnumpy(E_max)
        E_min = cp.asnumpy(E_min)
        e = cp.asnumpy(e)
        Phi = cp.asnumpy(Phi)
        R2 = cp.asnumpy(R2)
    
    return E_max, E_min, e, Phi, R2


def process_channel(images_channel, angles, channel_name, pixel_indices=None):
    """
    Process one channel: calculate ellipse parameters for all (or test) pixels
    images_channel: array of shape (n_angles, height, width)
    """
    n_angles, height, width = images_channel.shape
    
    # Reshape to (n_angles, n_pixels)
    images_flat = images_channel.reshape(n_angles, -1)
    
    # Determine which pixels to process
    if pixel_indices is not None:
        images_flat = images_flat[:, pixel_indices]
        n_pixels = len(pixel_indices)
    else:
        n_pixels = images_flat.shape[1]
    
    print(f"\nProcessing {channel_name} channel...")
    print(f"  Pixels to process: {n_pixels:,}")
    
    # Calculate E = sqrt(I) for all pixels at all angles
    print(f"  Calculating E = sqrt(I)...")
    E_values = np.sqrt(images_flat.astype(np.float32))  # Shape: (n_angles, n_pixels)
    E_values = E_values.T  # Shape: (n_pixels, n_angles)
    
    # Process in batches to manage memory
    batch_size = 10000 if GPU_AVAILABLE else 5000
    n_batches = int(np.ceil(n_pixels / batch_size))
    
    print(f"  Processing in {n_batches} batches of {batch_size} pixels...")
    
    all_Emax = []
    all_Emin = []
    all_e = []
    all_Phi = []
    all_R2 = []
    
    for i in tqdm(range(n_batches), desc=f"  {channel_name} batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_pixels)
        
        E_batch = E_values[start_idx:end_idx]
        Emax, Emin, e, Phi, R2 = calculate_ellipse_params_gpu(E_batch, angles)
        
        all_Emax.append(Emax)
        all_Emin.append(Emin)
        all_e.append(e)
        all_Phi.append(Phi)
        all_R2.append(R2)
    
    # Concatenate results
    all_Emax = np.concatenate(all_Emax)
    all_Emin = np.concatenate(all_Emin)
    all_e = np.concatenate(all_e)
    all_Phi = np.concatenate(all_Phi)
    all_R2 = np.concatenate(all_R2)
    
    # Create pixel coordinates
    if pixel_indices is not None:
        y_coords = pixel_indices // width
        x_coords = pixel_indices % width
    else:
        y_coords, x_coords = np.divmod(np.arange(n_pixels), width)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'Emax': all_Emax,
        'Emin': all_Emin,
        'eccentricity': all_e,
        'Phi': all_Phi,
        'R2': all_R2
    })
    
    return df


def main():
    start_time = time.time()
    
    # Load all images
    angles = list(range(0, 370, 10))
    images_R, images_G, images_B, angles = load_all_images(angles)
    
    # Determine which pixels to process
    if TEST_MODE:
        # Select first TEST_PIXELS pixels
        pixel_indices = np.arange(TEST_PIXELS)
        print(f"\nTEST MODE: Processing first {TEST_PIXELS} pixels")
    else:
        pixel_indices = None
        print(f"\nFULL MODE: Processing all {ROI_width * ROI_height:,} pixels")
    
    # Create output directory
    os.makedirs("./Practica7_resultados", exist_ok=True)
    
    # Process each channel
    df_R = process_channel(images_R, angles, 'Red', pixel_indices)
    df_G = process_channel(images_G, angles, 'Green', pixel_indices)
    df_B = process_channel(images_B, angles, 'Blue', pixel_indices)
    
    # Save results
    output_R = "./Practica7_resultados/ellipses_red.csv"
    output_G = "./Practica7_resultados/ellipses_green.csv"
    output_B = "./Practica7_resultados/ellipses_blue.csv"
    
    print(f"\nSaving results...")
    df_R.to_csv(output_R, index=False)
    print(f"  Red channel saved to: {output_R}")
    
    df_G.to_csv(output_G, index=False)
    print(f"  Green channel saved to: {output_G}")
    
    df_B.to_csv(output_B, index=False)
    print(f"  Blue channel saved to: {output_B}")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}\n")
    
    # Display sample results
    print("\nSample results (first 10 pixels, Red channel):")
    print(df_R.head(10))
    
    print("\nStatistics (Red channel):")
    print(df_R.describe())


if __name__ == "__main__":
    main()
