import configparser
import os
import subprocess
from pathlib import Path


# ==========================================================
# User settings
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
BASE_CONFIG = SCRIPT_DIR / "config_raw_data_chunkfdk.cfg"
TEMP_CONFIG = SCRIPT_DIR / "__cgls_raw_bag_config.cfg"

ITERATIONS = 20
BAG_NUMBERS = range(1, 7)

DATA_ROOT = SCRIPT_DIR.parents[2] / "real_bag_ct"
AIR_FILE = DATA_ROOT / "BS0.131_RPM120_Gain10_AirScan.dat"
OUTPUT_ROOT = SCRIPT_DIR.parent / "Real_Bag_CT_BS0.131_RPM120_Gain10"


def config_path(path: Path) -> str:
    """Return a path relative to SRC, as expected by the pipeline config."""
    return os.path.relpath(path, SCRIPT_DIR)


def run_bag(bag_number: int) -> None:
    bag_file = DATA_ROOT / f"BS0.131_RPM120_Gain10_Bag{bag_number}.dat"
    if bag_number == 6 and not bag_file.is_file():
        bag_file = DATA_ROOT / "BS0.131_RPM120_Gain10_Bag6_CTP.dat"
    if not bag_file.is_file():
        raise FileNotFoundError(f"Object scan not found: {bag_file}")

    bag_root = OUTPUT_ROOT / f"Bag{bag_number}" / f"CGLS_{ITERATIONS}"
    recon_dir = bag_root / "Recon"
    proj_dir = bag_root / "Projections"

    for directory in (
        recon_dir,
        proj_dir,
        recon_dir / "visualization",
        proj_dir / "visualization",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cfg = configparser.ConfigParser()
    cfg.read(BASE_CONFIG)

    data_config = cfg["data-preprocessing"]
    pipeline_config = cfg["pipeline"]
    visualization_config = cfg["visualization"]

    data_config["object_file"] = config_path(bag_file)
    data_config["air_files"] = repr([config_path(AIR_FILE)])
    data_config["conveyor_speed"] = "0.131"
    data_config["revolution_speed"] = "120"

    pipeline_config["init_algorithm"] = "ChunkFDK"
    pipeline_config["recon_algorithm"] = "CGLS"
    pipeline_config["niter"] = str(ITERATIONS)
    pipeline_config["processed_save_path"] = config_path(
        proj_dir / "processed_data.npz"
    )
    pipeline_config["recon_save_path"] = config_path(recon_dir)

    visualization_config["recon_output_dir"] = config_path(
        recon_dir / "visualization"
    )
    visualization_config["proj_output_dir"] = config_path(
        proj_dir / "visualization"
    )

    with TEMP_CONFIG.open("w") as config_file:
        cfg.write(config_file)

    print("=" * 70)
    print(f"Running Bag {bag_number} with CGLS ({ITERATIONS} iterations)")
    print("=" * 70)
    subprocess.run(
        ["python", str(MAIN_SCRIPT), "--config", str(TEMP_CONFIG)],
        cwd=SCRIPT_DIR,
        check=True,
    )


def main() -> None:
    if not AIR_FILE.is_file():
        raise FileNotFoundError(f"Air scan not found: {AIR_FILE}")

    try:
        for bag_number in BAG_NUMBERS:
            run_bag(bag_number)
    finally:
        TEMP_CONFIG.unlink(missing_ok=True)

    print("\nFinished all raw bag experiments.")


if __name__ == "__main__":
    main()