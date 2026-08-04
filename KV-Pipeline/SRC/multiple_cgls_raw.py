import os
import subprocess
import configparser

# ============================================================
# User Settings
# ============================================================

MAIN_SCRIPT = "main.py"
# BASE_CONFIG = "config_raw_data_chunkfdk.cfg"
BASE_CONFIG = "config_raw_data.cfg"
TEMP_CONFIG = "_fdk_temp_config.cfg" #! Change this

# CGLS iterations to test
ITERATIONS = list(range(0, 21))
# ITERATIONS.reverse()

# ============================================================

for niter in ITERATIONS:

    print("=" * 70)
    print(f"Running CGLS with {niter} iterations")
    print("=" * 70)

    # -------------------------------
    # Read original config
    # -------------------------------
    cfg = configparser.ConfigParser()
    cfg.optionxform = str          # preserve case
    cfg.read(BASE_CONFIG)

    # -------------------------------
    # Modify only required parameters
    # -------------------------------

    # Update CGLS iterations
    cfg["pipeline"]["niter"] = str(niter)

    # -----------------------------
    # Construct experiment folder
    # -----------------------------
    exp_root = (
        # f"../May_26th_BS0.09_80RPM_Iso1.3mm_Algo_ChunkFDK/"
        f"../May_26th_BS0.09_80RPM_Iso1.0mm_Algo_FDK/"
        f"CGLS_{niter}"
    )

    recon_dir = os.path.join(exp_root, "Recon")
    proj_dir = os.path.join(exp_root, "Projections")

    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(os.path.join(recon_dir, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "visualization"), exist_ok=True)

    # -----------------------------
    # Update every save path
    # -----------------------------
    cfg["pipeline"]["processed_save_path"] = os.path.join(
        proj_dir,
        "processed_data.npz",
    )

    cfg["pipeline"]["recon_save_path"] = recon_dir

    cfg["visualization"]["recon_output_dir"] = os.path.join(
        recon_dir,
        "visualization",
    )

    cfg["visualization"]["proj_output_dir"] = os.path.join(
        proj_dir,
        "visualization",
    )

    # -------------------------------
    # Write temporary config
    # -------------------------------
    with open(TEMP_CONFIG, "w") as f:
        cfg.write(f)

    # -------------------------------
    # Execute pipeline
    # -------------------------------
    subprocess.run(
        [
            "python",
            MAIN_SCRIPT,
            "--config",
            TEMP_CONFIG,
        ],
        check=True,
    )

# Cleanup
if os.path.exists(TEMP_CONFIG):
    os.remove(TEMP_CONFIG)

print("\nFinished all runs.")