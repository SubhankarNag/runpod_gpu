import copy
import numpy as np
import tigre
import tigre.algorithms as algs

# >>> CHANGE 1: Added imports for GPU discovery and parallel thread management
import tigre.utilities.gpu as gpu_utils
from concurrent.futures import ThreadPoolExecutor


# >>> CHANGE 2: Extracted the chunk processing logic out of the sequential loop 
# into a worker function to enable asynchronous, multi-threaded execution across GPUs
def _process_chunk_worker(args):
    """Worker function to process a single chunk on an assigned GPU."""
    (n, proj_chunk, angles_chunk, arr_chunk, geo_base, 
     offOrigin_z, start, end, center_phys, gpu_id, verbose) = args

    # >>> CHANGE 3: Explicitly bind this thread/chunk to a specific GPU ID
    gpuids = gpu_utils.GpuIds()
    gpuids.devices = [gpu_id]

    # Localized geometry setup
    geo_chunk = copy.deepcopy(geo_base)
    chunk_len = end - start
    geo_chunk.nVoxel[0] = chunk_len
    geo_chunk.sVoxel[0] = chunk_len * geo_base.dVoxel[0]

    chunk_center_phys = ((start + end) / 2.0) * geo_base.dVoxel[0]
    diff = center_phys - chunk_center_phys

    geo_chunk.offOrigin = np.zeros((arr_chunk.shape[0], 3))
    geo_chunk.offOrigin[:, 2] = offOrigin_z
    geo_chunk.offOrigin[:, 0] = arr_chunk - diff

    # >>> CHANGE 4: Removed unnecessary dummy array memory allocations 
    # (zero_proj, flatten_detector, unflatten_detector) that were causing CPU/GPU overhead

    if verbose:
        print(f"[GPU {gpu_id}] Reconstructing chunk {n+1} (Indices: {start} -> {end})...")

    # >>> CHANGE 5: Passed `gpuids=gpuids` into `algs.fdk` to force execution on the assigned GPU
    recon_chunk = algs.fdk(
        proj_chunk, 
        geo_chunk, 
        angles_chunk, 
        filter="hann", 
        gpuids=gpuids, 
        verbose=False
    )

    return start, end, recon_chunk


# >>> CHANGE 6: Added `device_ids` parameter to accept multiple target GPUs
def chunked_fdk(proj, geo, angles, chunk_size_idx=44, overlap_size_idx=24, device_ids=None, verbose=False):
    """
    Multi-GPU accelerated chunked FDK reconstruction.
    """
    # >>> CHANGE 7: Auto-detect available system GPUs if `device_ids` is not provided
    if device_ids is None:
        gpu_names = gpu_utils.getGpuNames()
        device_ids = list(range(len(gpu_names))) if gpu_names else [0]

    num_gpus = len(device_ids)
    step_idx = chunk_size_idx - overlap_size_idx
    total_len_idx = int(geo.nVoxel[0])

    starts_idx = np.arange(0, total_len_idx, step_idx)
    ends_idx = starts_idx + chunk_size_idx

    valid = ends_idx <= total_len_idx
    starts_idx = starts_idx[valid]
    ends_idx = ends_idx[valid]

    arr = geo.offOrigin[:, 0]
    center_phys = geo.sVoxel[0] / 2.0

    low_phys = center_phys - (ends_idx * geo.dVoxel[0]) - 5.0
    high_phys = center_phys - (starts_idx * geo.dVoxel[0]) + 5.0

    masks = (arr[None, :] >= low_phys[:, None] - 1e-9) & (arr[None, :] <= high_phys[:, None] + 1e-9)
    valid_masks = masks.any(axis=1)
    
    masks = masks[valid_masks]
    starts_idx = starts_idx[valid_masks]
    ends_idx = ends_idx[valid_masks]

    scene_shape = tuple(geo.nVoxel)
    combined = np.zeros(scene_shape, dtype=np.float32)
    counts = np.zeros(scene_shape, dtype=np.int32)

    print(f"Starting Multi-GPU ChunkFDK: {len(starts_idx)} chunks across GPUs {device_ids}")

    # >>> CHANGE 8: Pre-build worker task arguments and assign GPUs in a round-robin fashion
    tasks = []
    for n in range(len(starts_idx)):
        m = masks[n]
        gpu_id = device_ids[n % num_gpus]  # Round-robin scheduling across GPUs
        
        task_args = (
            n,
            proj[m],
            angles[m],
            arr[m],
            geo,
            geo.offOrigin[m, 2],
            starts_idx[n],
            ends_idx[n],
            center_phys,
            gpu_id,
            verbose
        )
        tasks.append(task_args)

    # >>> CHANGE 9: Replaced sequential `for` loop with `ThreadPoolExecutor` 
    # to process chunks simultaneously across worker GPUs
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        results = executor.map(_process_chunk_worker, tasks)

        # Collect and sum results into host memory as they finish
        for start, end, recon_chunk in results:
            combined[start:end, :, :] += recon_chunk
            counts[start:end, :, :] += 1

    print("Stitching overlapping chunks...")
    final_array = np.divide(combined, counts, out=np.zeros_like(combined), where=counts > 0)

    return final_array