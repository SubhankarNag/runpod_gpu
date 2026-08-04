"""
Main Pipeline Orchestrator
Ties together configuration loading, data processing (or simulation), TIGRE geometry 
initialization, matrix flattening, and calls the reconstruction algorithms.
"""
import os
import numpy as np
import argparse
import tigre
from math import ceil, floor
from tigre.utilities.flatten_detector import flatten_detector, unflatten_detector

from pipeline_config import load_config
from tigre.utilities.working_geometry import CTGeometry
from data_processing import process_raw_dat, save_image, make_gif
from reconstruction import all_algorithms

def main():
    # Setup argparse for config override
    parser = argparse.ArgumentParser(description="CT Reconstruction Pipeline")
    parser.add_argument('--config', type=str, default='config.cfg', help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. GPU Configuration
    # Supports passing a list/tuple of GPUs (e.g. [0,1]) or a single string
    gpu_config = config['system']['gpu_ids']
    if isinstance(gpu_config, (tuple, list)):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_config))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_config)

    # 2. Extract Data or Run Simulation
    processed_path = config['pipeline']['processed_save_path']
    
    # Check if we should bypass computation and load cached numpy arrays
    if config['pipeline']['save_processed_arrays'] and os.path.exists(processed_path) and config['visualization'].get('load_from_saved_path', False):
        print(f"Loading cached arrays from {processed_path}...")
        data = np.load(processed_path)
        projections = data['corrected_pixels']
        angles = data['angles']
        displacement = data['displacement']
    else:
        # Branch A: Simulation Mode
        if config['data-preprocessing']['data_mode'] == 'simulated':
            print("Running in simulated mode...")
            start_a, end_a, step_a = config['data-preprocessing']['sim_angles']
            scene_length_along_belt = config['geometry']['svoxel_x']
            belt_speed = config['data-preprocessing']['conveyor_speed']
            belt_speed = belt_speed * 1000 # conversion into mm/sec
            rev_rpm = config['data-preprocessing']['revolution_speed']

            num_projs = (360 / step_a) * (scene_length_along_belt / belt_speed) * (rev_rpm / 60)
            angles = step_a * np.arange(num_projs)
            angles = (- angles) % 360
            angles = angles * (np.pi / 180.0)

            time_btn_adj_proj = (step_a / 360) / (rev_rpm / 60)
            displacement = np.arange(num_projs) * belt_speed * time_btn_adj_proj
            center = displacement[len(displacement) // 2]
            displacement = displacement - center
            
            # Init temporary geometry to compute the forward projection
            temp_geo = CTGeometry(config['geometry'])
            temp_geo.update_for_angles(len(angles), config['geometry']['shift_z'], -displacement)

            # Internally update TIGRE geometry states via flattening logic
            zero_proj = np.zeros((angles.shape[0], temp_geo.nDetectorCurved[0], temp_geo.nDetectorCurved[1]))
            zero_flattened_proj = flatten_detector(zero_proj, temp_geo)
            _ = unflatten_detector(zero_flattened_proj, temp_geo)
            
            print(f"Loading simulated scene from {config['data-preprocessing']['simulated_scene_path']}...")
            actual_fov = np.load(config['data-preprocessing']['simulated_scene_path'])

            # Plot input simulated scene if requested
            if config['visualization'].get('plot_simulated_scene', False):
                scene_gif = os.path.join(config['visualization']['scene_output_dir'], "simulated_scene.gif")
                os.makedirs(os.path.dirname(scene_gif), exist_ok=True)
                print(f"Saving simulated scene GIF to {scene_gif}...")
                tigre.plotimg(actual_fov, dim="Z", savegif=scene_gif)
                
            print("Forward projecting simulated scene...")
            flattened_proj = tigre.Ax(actual_fov, temp_geo, angles)
            
            # Unflatten to mimic raw detector output shape exactly
            projections = unflatten_detector(flattened_proj, temp_geo)

            # Extract simulated projections to PNG/GIF
            if config['visualization'].get('save_projection_images', False) or config['visualization'].get('save_projection_gif', False):
                print("Saving simulated projections images and GIF...")
                frames_init = []
                global_min, global_max = float("inf"), float("-inf")
                freq = config['visualization'].get('save_frequency', 50)
                out_dir = config['visualization']['proj_output_dir']
                
                for i in range(projections.shape[0]):
                    proj = projections[i]
                    global_min = min(global_min, proj.min())
                    global_max = max(global_max, proj.max())
                    
                    if i % freq == 0:
                        frames_init.append(proj)
                        if config['visualization'].get('save_projection_images', True):
                            angle_deg = angles[i] * (180.0 / np.pi)
                            filepath = os.path.join(out_dir, "simulated_projections", f"proj_{i}_{angle_deg:.2f}.png")
                            save_image(proj, filepath)
                
                if config['visualization'].get('save_projection_gif', True) and frames_init:
                    gif_path = os.path.join(out_dir, "simulated_projections.gif")
                    make_gif(frames_init, global_min, global_max, gif_path, fps=config['visualization']['gif_fps'])

            # Reverse Angle alignment for simulation (to ensure identical behavior with raw)
            angles = np.roll(angles, -config['pipeline']['roll_angles'])

        # Branch B: Raw Physical Scanner Mode
        elif config['data-preprocessing']['data_mode'] == 'raw_dat':
            projections, angles, displacement = process_raw_dat(config['data-preprocessing'], config['visualization'])
        
        # Cache processed results for future runs
        if config['pipeline']['save_processed_arrays']:
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            np.savez_compressed(processed_path, corrected_pixels=projections, angles=angles, displacement=displacement)

    # 3. Apply manual horizontal rotation offset
    angles = np.roll(angles, config['pipeline']['roll_angles'])

    # 4. Centering of displacement
    center = displacement[len(displacement) // 2]
    displacement = displacement - center

    # 5. Calculate Length along the conveyor belt using displacements
    scene_length_along_belt = ceil(abs(displacement[-1] - displacement[0]))
    config['geometry']['svoxel_x'] = min(config['geometry']['svoxel_x'], scene_length_along_belt)

    # 6. Initialize Core TIGRE Geometry for Reconstruction
    geo = CTGeometry(config['geometry'])
    geo.update_for_angles(len(angles), config['geometry']['shift_z'], -displacement)

    # Internal TIGRE flatten trigger logic to populate missing parameters
    zero_proj = np.zeros((angles.shape[0], geo.nDetectorCurved[0], geo.nDetectorCurved[1]))
    zero_flattened_proj = flatten_detector(zero_proj, geo)
    _ = unflatten_detector(zero_flattened_proj, geo)

    # 7. Format and Flatten Projections for TIGRE backend
    print(f"Flattening projection array of shape {projections.shape}...")
    proj_flat = flatten_detector(projections, geo).astype(np.float32)

    # 8. Run Reconstructions
    recon_algo = config['pipeline']['recon_algorithm']
    init_algo = config['pipeline']['init_algorithm']
    out_dir = config['pipeline']['recon_save_path']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reconsturction of Scene with shape ({geo.nVoxel[0]}, {geo.nVoxel[1]}, {geo.nVoxel[2]})")
    print(f"Starting Initial {init_algo} reconstruction...")
    init_vol = all_algorithms(
        geo, angles, proj_flat, init_algo, out_dir, 
        config['visualization']['save_recon_gif']
    )
    
    # If the user requested a different algorithm (like CGLS), pass the FDK result as the initial state
    if recon_algo != init_algo:
        print(f"Starting Primary {recon_algo} reconstruction...")
        final_vol = all_algorithms(
            geo, angles, proj_flat, recon_algo, out_dir, 
            config['visualization']['save_recon_gif'], init=init_vol,
            niter=config['pipeline']['niter']
        )
    else:
        final_vol = init_vol

    # 9. Final Output Alignments
    np.save(os.path.join(out_dir, "final_volume.npy"), final_vol)
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()