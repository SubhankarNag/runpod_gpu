"""
Data Processing Module
Handles reading raw .dat files, aligning Air and Object scans based on encoder ticks,
computing the log projections, tracking physical displacement, and exporting intermediate debug images/GIFs.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont
import os

# --- HELPER LOGIC ---
def make_int_if_possible(x):
    """Converts a float to an int if there is no fractional part."""
    return int(x) if x.is_integer() else x

def log_done(wedge, air):
    """Computes the attenuation projection using Beer-Lambert Law: -log(I/I0)"""
    return np.log(air / wedge)

def making_projection(image):
    """Restructures a flat raw string of pixels into the 32x480 scintillator format."""
    avg = np.mean(image, axis=0)
    image_intermediate = np.split(avg, 30)
    one = image_intermediate[0].reshape(32, 16)
    for i in range(1, 30):
        one1 = image_intermediate[i].reshape(32, 16)
        one = np.concatenate((one, one1), axis=1)
    return one

# --- VISUALIZATION UTILS ---
def save_image(array, filepath):
    """Saves a 2D numpy array as a pseudo-colored image."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.imsave(filepath, array, cmap="jet")

def plot_field(field, name, path):
    """Plots a 1D array as a line graph (used for debugging machine ticks)."""
    plt.figure(figsize=(8,4))
    plt.plot(field)
    plt.title(name)
    plt.xlabel("View index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def generate_debug_plots(metadata, ticks, output_dir, start_view, end_view, displacement):
    """Generates graphs for displacement and timing to debug scanner stability."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_1us = (
        (metadata[:, 1].astype(np.uint64) << 48) |
        (metadata[:, 2].astype(np.uint64) << 32) |
        (metadata[:, 3].astype(np.uint64) << 16) |
        metadata[:, 4].astype(np.uint64)
    )
    
    # Applying slice limits directly to visualization for consistency
    t_slice = timestamp_1us[start_view:end_view]
    t_diff = np.diff(t_slice)
    
    plot_field(t_slice, "timestamp_1us", os.path.join(output_dir, "timestamp_1us.png"))
    plot_field(t_diff, "timestamp_1us_diff", os.path.join(output_dir, "timestamp_1us_diff.png"))
    
    if len(displacement) > 0:
        plot_field(displacement, "displacement", os.path.join(output_dir, "displacement.png"))
        
    plot_field(ticks, "encoder_ticks", os.path.join(output_dir, "object_tick.png"))

def make_gif(frames, global_min, global_max, filepath, fps=5):
    """Aggregates a list of 2D numpy arrays into an animated GIF."""
    cmap = plt.get_cmap("jet")
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    gif_frames = []
    HEADER = 20
    
    for i, proj in enumerate(frames):
        # Normalize and colorize
        temp = (proj - global_min) / (global_max - global_min + 1e-8)
        temp = np.clip(temp, 0, 1)
        temp_colored = (cmap((temp * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
        
        # Add a white header for text
        h, w, _ = temp_colored.shape
        canvas = np.ones((h + HEADER, w, 3), dtype=np.uint8) * 255
        canvas[HEADER:, :, :] = temp_colored
        
        # Write Z-index onto the frame
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        text = f"Z = {i * 50}"
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((w - (bbox[2] - bbox[0])) // 2, (HEADER - (bbox[3] - bbox[1])) // 2), text, fill=(0, 0, 0), font=font)
        
        gif_frames.append(np.array(img))

    imageio.mimsave(filepath, gif_frames, fps=fps, loop=0)

# --- CORE PROCESSING ---
def extraction(fname, config_data, start=0, end=None):
    """Reads raw 16-bit binary .dat files, isolates pixel data, and calculates physical displacement."""
    META = config_data['meta_size']
    PIXELS = config_data['rows'] * config_data['cols']
    RECORD = META + PIXELS

    raw = np.fromfile(fname, dtype=np.uint16)
    views = raw.size // RECORD
    
    if end is None:
        end = views
    
    records = raw.reshape(views, RECORD)
    metadata = records[:, :META]
    pixels = records[:, META:]
    ticks = metadata[:, 13].astype(np.int32)
    
    # Extract timestamp and compute displacement exactly like input_conversion.py
    timestamp_1us = (
        (metadata[:, 1].astype(np.uint64) << 48) |
        (metadata[:, 2].astype(np.uint64) << 32) |
        (metadata[:, 3].astype(np.uint64) << 16) |
        metadata[:, 4].astype(np.uint64)
    )
    
    # Read conveyor_speed securely from config
    speed = config_data['conveyor_speed']
    speed = speed * 1000 # conversion into mm/sec
    
    t0 = timestamp_1us[0] if len(timestamp_1us) > 0 else 0
    displacement = speed * (timestamp_1us - t0) * 1e-6
    
    return metadata, pixels[start:end], ticks[start:end], displacement[start:end]

def get_valid_indices(ticks, ticks_per_proj):
    """Filters out erroneous encoder ticks."""
    diff = np.diff(ticks)
    valid = [i for i in range(1, len(ticks)-1) if diff[i-1] == -ticks_per_proj and diff[i] == -ticks_per_proj]
    return np.array(valid)

def process_raw_dat(config_data, config_viz):
    """
    Main orchestration function for dat files. 
    Averages Air scans, aligns them to Object scans via encoder ticks, 
    and returns final cleaned projection matrices and rotation angles.
    """
    print(f"Extracting {config_data['object_file']}...")
    # Configurable Slicing logic
    start = config_data.get('start_view', 0)
    end = config_data.get('end_view', None)
    obj_meta, pixels, ticks, obj_disp = extraction(config_data['object_file'], config_data, start, end)

    if config_viz['generate_debug_plots']:
        generate_debug_plots(
            obj_meta, ticks, os.path.join(config_viz['proj_output_dir'], "debug"), 
            start, end, obj_disp
        )

    # Automatically defaults to 8
    ticks_per_proj = config_data.get('ticks_per_projection', 8)
    
    valid_obj_idx = get_valid_indices(ticks, ticks_per_proj)
    # print(valid_obj_idx) #! changed 
    pixels_obj = pixels[valid_obj_idx]
    ticks_obj = ticks[valid_obj_idx]
    disp_obj = obj_disp[valid_obj_idx]

    T_MAX = config_data['t_max']
    def ticks_to_angle_idx(t):
        return ((T_MAX - t) // ticks_per_proj).astype(int)
    def idx_to_angle(idx):
        return (idx * ticks_per_proj) // 32
    
    # Group Air Averages across ALL provided air scans
    air_groups = {}
    for air_file in config_data['air_files']:
        print(f"Extracting {air_file}...")
        _, air_pixels, air_ticks, _ = extraction(air_file, config_data)
        
        valid_air_idx = get_valid_indices(air_ticks, ticks_per_proj)
        air_pixels = air_pixels[valid_air_idx]
        air_ticks = air_ticks[valid_air_idx]
        
        for i, idx in enumerate(ticks_to_angle_idx(air_ticks)):
            if idx not in air_groups: 
                air_groups[idx] = []
            air_groups[idx].append(air_pixels[i])

    # Average the grouped air frames
    air_avg = {idx: np.mean(imgs, axis=0) for idx, imgs in air_groups.items()}

    obj_angle_idx = ticks_to_angle_idx(ticks_obj)

    corrected_pixels, angles, corrected_displacement = [], [], []
    frames_init = []
    global_min, global_max = float("inf"), float("-inf")
    
    print("Computing log projections...")
    for i, idx in enumerate(obj_angle_idx):
        if idx in air_avg:
            I = making_projection(pixels_obj[i].reshape(1, -1))
            I0 = making_projection(air_avg[idx].reshape(1, -1))

            # Apply Beer-Lambert law
            proj = log_done(I, I0)
            proj[proj < 0] = 0

            global_min = min(global_min, proj.min())
            global_max = max(global_max, proj.max())

            # Save diagnostic projection images
            if idx % config_viz['save_frequency'] == 0:
                frames_init.append(proj)
                if config_viz['save_projection_images']:
                    save_image(proj, os.path.join(config_viz['proj_output_dir'], "projections", f"proj_{i}_{idx_to_angle(idx)}.png"))

            corrected_pixels.append(proj)
            angles.append(idx_to_angle(idx))
            corrected_displacement.append(disp_obj[i])

    if config_viz['save_projection_gif'] and frames_init:
        make_gif(frames_init, global_min, global_max, os.path.join(config_viz['proj_output_dir'], "projections.gif"), fps=config_viz['gif_fps'])

    corrected_pixels = np.stack(corrected_pixels, axis=0)
    corrected_displacement = np.array(corrected_displacement)
    
    # Process angles to handle full 360 wrap-arounds
    angles = (- np.array(angles)) % 360
    
    final_angles = angles * (np.pi / 180) # Convert to radians
    final_projs = corrected_pixels
    final_disp = corrected_displacement

    return final_projs, final_angles, final_disp