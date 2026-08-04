"""
Chunked Reconstruction Module
Implements the overlapping chunks FDK algorithm to reconstruct infinitely long 
conveyor belt scans by processing data in localized windows (chunks) and 
blending the overlapping regions.
"""
import copy
import numpy as np
import tigre
import tigre.algorithms as algs
from tigre.utilities.flatten_detector import flatten_detector, unflatten_detector

def chunked_fdk(proj, geo, angles, chunk_size_idx=44, overlap_size_idx=24, verbose=False):
    """
    Reconstructs a long scan by dividing it into overlapping chunks along the 
    conveyor belt axis (TIGRE's Z-axis, which maps to geo.nVoxel[0]).
    
    Args:
        proj: Flat projection array
        geo: Base TIGRE geometry covering the total scan length
        angles: Array of projection angles
        chunk_size_idx: Number of voxels along the belt axis per chunk
        overlap_size_idx: Number of overlapping voxels between consecutive chunks
        verbose: Print progress
        
    Returns:
        final_array: The fully stitched 3D reconstructed volume
    """
    # 1. Define window steps and indices
    step_idx = chunk_size_idx - overlap_size_idx
    total_len_idx = int(geo.nVoxel[0])
    
    starts_idx = np.arange(0, total_len_idx, step_idx)
    ends_idx = starts_idx + chunk_size_idx
    
    # Keep only windows that fit entirely within the total length bounds
    valid = ends_idx <= total_len_idx
    starts_idx = starts_idx[valid]
    ends_idx = ends_idx[valid]
    
    # 2. Extract physical displacement array
    # In the KV-Pipeline, geo.offOrigin[:, 0] represents physical machine displacement
    arr = geo.offOrigin[:, 0]
    
    # Calculate global physical center of the reconstructed field
    center_phys = geo.sVoxel[0] / 2.0
    
    # 3. Create mapping masks for projections
    # We find which projections physically fall into each chunk's bounds.
    # A +/- 5 unit physical margin is added to ensure complete ray coverage.
    low_phys = center_phys - (ends_idx * geo.dVoxel[0]) - 5.0
    high_phys = center_phys - (starts_idx * geo.dVoxel[0]) + 5.0
    
    # Vectorized mask computation: check if projection position 'arr' is within bounds
    masks = (arr[None, :] >= low_phys[:, None] - 1e-9) & (arr[None, :] <= high_phys[:, None] + 1e-9)
    
    # Filter out any chunks that caught 0 projections (prevents empty array crashes)
    valid_masks = masks.any(axis=1)
    masks = masks[valid_masks]
    starts_idx = starts_idx[valid_masks]
    ends_idx = ends_idx[valid_masks]
    
    # 4. Initialize the global combined array and counts for averaging
    scene_shape = tuple(geo.nVoxel)
    combined = np.zeros(scene_shape, dtype=np.float32)
    counts = np.zeros(scene_shape, dtype=np.int32)
    
    print(f"Starting ChunkFDK: {len(starts_idx)} chunks to process.")
    
    # 5. Process each chunk independently
    for n in range(len(starts_idx)):
        m = masks[n]
        start = starts_idx[n]
        end = ends_idx[n]
        
        proj_chunk = proj[m]
        angles_chunk = angles[m]
        arr_chunk = arr[m]
        
        # Clone global geometry to create a localized sub-geometry
        geo_chunk = copy.deepcopy(geo)
        geo_chunk.nVoxel[0] = chunk_size_idx
        geo_chunk.sVoxel[0] = chunk_size_idx * geo.dVoxel[0]
        
        # Trigger internal TIGRE C++ memory allocations for the new detector/geometry shape
        zero_proj = np.zeros((angles_chunk.shape[0], geo_chunk.nDetectorCurved[0], geo_chunk.nDetectorCurved[1]))
        zero_flattened_proj = flatten_detector(zero_proj, geo_chunk)
        _ = unflatten_detector(zero_flattened_proj, geo_chunk)
        
        # Center the chunk physically relative to the global scene
        chunk_center_phys = ((start + end) / 2.0) * geo.dVoxel[0]
        diff = center_phys - chunk_center_phys
        
        # Adjust offsets: X-axis shifts relative to chunk center, Z-axis preserves OO' shift
        geo_chunk.offOrigin = np.zeros((arr_chunk.shape[0], 3))
        geo_chunk.offOrigin[:, 2] = geo.offOrigin[m, 2] 
        geo_chunk.offOrigin[:, 0] = arr_chunk - diff
        
        if verbose:
            print(f"Reconstructing chunk {n+1}/{len(starts_idx)} (Indices: {start} to {end})...")
            
        # Execute standard analytical FDK on the small overlapping chunk
        recon_chunk = algs.fdk(proj_chunk, geo_chunk, angles_chunk, filter="hann", verbose=verbose)
        
        # Add the reconstructed chunk back into the global volume
        combined[start:end, :, :] += recon_chunk
        counts[start:end, :, :] += 1
        
    # 6. Stitching: Average out the overlapping regions
    print("Stitching overlapping chunks...")
    final_array = np.divide(combined, counts, out=np.zeros_like(combined), where=counts > 0)
    
    return final_array