import configparser
import subprocess
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

MAIN_SCRIPT = "main.py"
# BASE_CFG = "config_simulated_chunkfdk.cfg"
BASE_CFG = "config_simulated_small.cfg"
TMP_CFG = "__fdk_tmp_config.cfg"

# CGLS iterations to test
ITERATIONS = [1,5,7,10,12,15,20]

# ==========================================================

for niter in ITERATIONS:

    print("=" * 70)
    print(f"Running CGLS with {niter} iterations")
    print("=" * 70)

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(BASE_CFG)

    # ------------------------------------------------------
    # Update iteration
    # ------------------------------------------------------
    cfg["pipeline"]["niter"] = str(niter)

    # ------------------------------------------------------
    # Experiment directory
    # ------------------------------------------------------
    exp_root = (
        # f"../Simulated_Scene_Iso1.3mm_Algo_ChunkFDK_/"
        f"../Simulated_Scene_small_Iso1.0mm_Algo_FDK_multigpu_/"
        f"CGLS_{niter}"
    )

    recon_dir = os.path.join(exp_root, "Recon")
    proj_dir = os.path.join(exp_root, "Projections")
    scene_vis_dir = os.path.join(exp_root, "visualization")

    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(os.path.join(recon_dir, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "visualization"), exist_ok=True)
    os.makedirs(scene_vis_dir, exist_ok=True)

    # ------------------------------------------------------
    # Update config paths
    # ------------------------------------------------------
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

    cfg["visualization"]["scene_output_dir"] = scene_vis_dir

    # ------------------------------------------------------
    # Write temporary config
    # ------------------------------------------------------
    with open(TMP_CFG, "w") as f:
        cfg.write(f)

    # ------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------
    subprocess.run(
        [
            "python",
            MAIN_SCRIPT,
            "--config",
            TMP_CFG,
        ],
        check=True,
    )

# Cleanup
if os.path.exists(TMP_CFG):
    os.remove(TMP_CFG)

print("\nFinished all experiments.")