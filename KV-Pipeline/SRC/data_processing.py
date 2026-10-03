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
import csv  # [NEW FIRMWARE] Needed to read separate metadata CSV

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

# [NEW FIRMWARE] Reads .dat files that have NO inline metadata.
# Metadata (ViewID, timestamps) comes from a separate CSV file.
# Used for BOTH object and air scans (3Oct2026 firmware onwards).
def extraction_new_firmware(fname, metadata_csv, config_data, start=0, end=None):
    """Reads new-firmware .dat files (pure pixel data, no inline metadata).
    Metadata is read from a separate CSV file."""
    PIXELS = config_data['rows'] * config_data['cols']

    raw = np.fromfile(fname, dtype=np.uint16)
    views = raw.size // PIXELS
    
    if end is None:
        end = views
    
    pixels = raw.reshape(views, PIXELS)
    
    # [NEW FIRMWARE] Read metadata from separate CSV
    view_ids = []
    timestamps_us = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_ids.append(int(row['ViewID']))
            timestamps_us.append(int(row['Timestamp(us)']))
    
    view_ids = np.array(view_ids, dtype=np.int32)
    timestamps_us = np.array(timestamps_us, dtype=np.uint64)
    
    # Compute displacement from timestamps (same logic as before)
    speed = config_data['conveyor_speed']
    speed = speed * 1000  # conversion into mm/sec
    t0 = timestamps_us[0] if len(timestamps_us) > 0 else 0
    displacement = speed * (timestamps_us - t0) * 1e-6
    
    # [NEW FIRMWARE] ViewID directly serves as the angular index (0-1439 range),
    # replacing the old encoder tick -> angle_idx conversion.
    return pixels[start:end], view_ids[start:end], displacement[start:end]

def get_valid_indices(ticks, ticks_per_proj):
    """Filters out erroneous encoder ticks."""
    diff = np.diff(ticks)
    valid = [i for i in range(1, len(ticks)-1) if diff[i-1] == -ticks_per_proj and diff[i] == -ticks_per_proj]
    return np.array(valid)

def process_raw_dat(config_data, config_viz):
    """
    Main orchestration function for dat files. 
    Averages Air scans, aligns them to Object scans via ViewID,
    and returns final cleaned projection matrices and rotation angles.
    """
    print(f"Extracting {config_data['object_file']}...")
    # Configurable Slicing logic
    start = config_data.get('start_view', 0)
    end = config_data.get('end_view', None)
    
    # [NEW FIRMWARE] Object scan uses new extraction (no inline metadata, separate CSV)
    metadata_csv = config_data['metadata_csv']
    pixels, obj_view_ids, obj_disp = extraction_new_firmware(
        config_data['object_file'], metadata_csv, config_data, start, end
    )

    # [NEW FIRMWARE] Debug plots: simplified since we no longer have inline metadata.
    # We plot ViewID and displacement directly.
    if config_viz['generate_debug_plots']:
        debug_dir = os.path.join(config_viz['proj_output_dir'], "debug")
        os.makedirs(debug_dir, exist_ok=True)
        plot_field(obj_view_ids, "ViewID", os.path.join(debug_dir, "view_id.png"))
        if len(obj_disp) > 0:
            plot_field(obj_disp, "displacement", os.path.join(debug_dir, "displacement.png"))

    # [NEW FIRMWARE] No tick-based filtering needed for object scan.
    # ViewID is clean and sequential - use all views directly.
    pixels_obj = pixels
    view_ids_obj = obj_view_ids
    disp_obj = obj_disp

    # [NEW FIRMWARE] ViewID directly maps to angle: angle_deg = ViewID * (360/1440) = ViewID / 4
    # This replaces the old ticks_to_angle_idx and idx_to_angle conversions.
    # 1440 views per revolution, each ViewID step = 0.25 degrees
    VIEWS_PER_REV = config_data.get('views_per_revolution', 1440)
    def view_id_to_angle(vid):
        return vid * 360.0 / VIEWS_PER_REV
    
    # Group Air Averages across ALL provided air scans
    # [NEW FIRMWARE] Air scan also uses new extraction with separate CSV metadata
    air_groups = {}
    air_files = config_data['air_files']
    air_metadata_csvs = config_data['air_metadata_csvs']
    
    for air_file, air_csv in zip(air_files, air_metadata_csvs):
        print(f"Extracting {air_file}...")
        # [NEW FIRMWARE] Air scan also has no inline metadata, uses CSV
        air_pixels, air_view_ids, _ = extraction_new_firmware(air_file, air_csv, config_data)
        
        # [NEW FIRMWARE] Group air frames by ViewID directly (no tick filtering needed)
        for i, vid in enumerate(air_view_ids):
            if vid not in air_groups: 
                air_groups[vid] = []
            air_groups[vid].append(air_pixels[i])

    # Average the grouped air frames
    air_avg = {vid: np.mean(imgs, axis=0) for vid, imgs in air_groups.items()}

    # [NEW FIRMWARE] Object's ViewID is directly used for alignment with air ViewID
    obj_angle_idx = view_ids_obj

    corrected_pixels, angles, corrected_displacement = [], [], []
    frames_init = []
    global_min, global_max = float("inf"), float("-inf")
    
    print("Computing log projections...")
    for i, vid in enumerate(obj_angle_idx):
        if vid in air_avg:
            # [NEW FIRMWARE] Both object and air pixels are already in 32x480 format
            I = pixels_obj[i].reshape(config_data['rows'], config_data['cols'])
            I0 = air_avg[vid].reshape(config_data['rows'], config_data['cols'])

            # Apply Beer-Lambert law
            proj = log_done(I, I0)
            proj[proj < 0] = 0

            global_min = min(global_min, proj.min())
            global_max = max(global_max, proj.max())

            angle_deg = view_id_to_angle(vid)
            # Save diagnostic projection images
            if vid % config_viz['save_frequency'] == 0:
                frames_init.append(proj)
                if config_viz['save_projection_images']:
                    save_image(proj, os.path.join(config_viz['proj_output_dir'], "projections", f"proj_{i}_{angle_deg:.2f}.png"))

            corrected_pixels.append(proj)
            angles.append(angle_deg)
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