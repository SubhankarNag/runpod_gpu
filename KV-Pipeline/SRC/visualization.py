"""
Post-Processing & Visualization Module
Provides tools to extract specific 2D slices from the reconstructed volume
and opens an interactive PyVista 3D render window. Applies artifact reduction filters.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import nibabel as nib
from scipy.ndimage import gaussian_filter, median_filter
from pipeline_config import load_config

# Attempt to configure PyVista for Jupyter compatibility (silent fail if in standard terminal)
try:
    pv.set_jupyter_backend("trame")
except ImportError:
    pass

def main_load_volume(path):
    """Loads the 3D volume intelligently based on file extension (.npy or .nii)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume not found at {path}")
        
    ext = os.path.splitext(path)[1]
    if ext == ".npy":
        return np.load(path)
    elif ext in [".nii", ".gz"]:
        nii = nib.load(path)
        return nii.get_fdata()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_volume_base(vol, align_orientation, max_clip):
    """Applies intensity clipping and standard axis orientation alignment."""
    if vol.ndim != 3:
        raise ValueError("Volume must be 3D")

    vol = vol.astype(np.float32)

    # Basic contrast clipping (removes extremely bright outlier artifacts)
    vol = np.clip(vol, 0, max_clip)
    vol[vol < 0] = 0

    # Physical alignment mapping (flips and transposes to standard views)
    if align_orientation:
        vol = vol.transpose((0, 2, 1))
        vol = vol[::-1, :, :]

    # Final axis rotation for correct top/front projections
    vol = vol.transpose((1, 2, 0))
        
    return vol

def save_2d_slices(vol, config_slices, save_dir, max_clip):
    """Iterates through user-requested slice lists and exports them as PNG images."""
    os.makedirs(save_dir, exist_ok=True)
    cmap = config_slices.get('cmap', 'jet')

    x_slices = config_slices.get('x_slices', [])
    y_slices = config_slices.get('y_slices', [])
    z_slices = config_slices.get('z_slices', [])

    def save_single_slice(slice_data, axis_name, idx):
        plt.figure()
        plt.title(f"{axis_name} slice = {idx}")
        plt.imshow(slice_data, cmap=cmap, vmin=0, vmax=max_clip)
        plt.colorbar()
        filepath = os.path.join(save_dir, f"{axis_name}_slice_{idx}.png")
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        print(f"Saved {filepath}")

    for x in x_slices:
        if 0 <= x < vol.shape[0]:
            save_single_slice(vol[x, :, :], "X", x)

    for y in y_slices:
        if 0 <= y < vol.shape[1]:
            save_single_slice(vol[:, y, :], "Y", y)

    for z in z_slices:
        if 0 <= z < vol.shape[2]:
            save_single_slice(vol[:, :, z], "Z", z)

def enhance_for_3d(vol):
    """
    Applies an aggressive combination of artifact reduction filters:
    1. Median: Removes star-like scattering
    2. Gaussian: Smooths ripple oscillations
    3. Edge-Preservation: Restores boundaries after blurring
    """
    print("Applying 3D post-processing filters (median, gaussian, edge-preservation)...")
    
    vol = median_filter(vol, size=5)
    vol = gaussian_filter(vol, sigma=0.5)

    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-8)

    smooth = gaussian_filter(vol, sigma=1.0)
    vol = np.where(np.abs(vol - smooth) < 0.1, smooth, vol)

    blur = gaussian_filter(vol, sigma=1.0)
    vol = np.clip(vol + 0.7 * (vol - blur), 0, 1)

    return vol

def create_3d_viewer(volume, apply_post_processing):
    """Initializes the PyVista volume plotter and renders the interactive scene."""
    print("Launching PyVista 3D Viewer...")
    
    if apply_post_processing:
        volume = enhance_for_3d(volume)
    else:
        # Standard normalization if filters are skipped
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin + 1e-8)

    grid = pv.wrap(volume)
    plotter = pv.Plotter(window_size=(1200, 900))

    # Transparency map: Removes empty air, solidifies denser objects
    opacity = [0.0, 0.01, 0.05, 0.2, 0.4, 0.7, 1.0]

    plotter.add_volume(
        grid,
        cmap="jet",
        opacity=opacity,
        shade=True,
    )

    plotter.enable_lightkit()
    plotter.add_axes()
    plotter.add_text("CT Volume Viewer", font_size=12)

    plotter.show()

def main():
    parser = argparse.ArgumentParser(description="CT Volume Visualization")
    parser.add_argument('--config', type=str, default='visualization_config.cfg', help='Path to visualization config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # just for exp:
    # config["slices"]["x_slices"] = list(range(0, ))


    vol_path = config['paths']['volume_path']
    save_dir = config['paths']['save_dir']
    max_clip = config['processing']['max_clip']
    align_orient = config['processing']['align_orientation']

    print(f"Loading volume from {vol_path}...")
    raw_vol = main_load_volume(vol_path)
    
    print("Applying base processing...")
    vol = process_volume_base(raw_vol, align_orient, max_clip)
    print(f"Processed Volume shape: {vol.shape}")

    # 1. Save Static 2D Slices
    print("Extracting 2D slices...")
    save_2d_slices(vol, config['slices'], save_dir, max_clip)

    # 2. Show Interactive 3D Viewer
    if config['3d_viewer']['show_3d']:
        create_3d_viewer(vol, config['3d_viewer']['apply_post_processing'])

if __name__ == "__main__":
    main()