# KV-Pipeline: CT Reconstruction & Visualization Framework

KV-Pipeline is a highly configurable, modular Computed Tomography (CT) reconstruction and visualization pipeline built on top of the TIGRE toolbox. It supports both:

- Physical machine data processing using raw `.dat` files
- Virtual 3D scene forward-projection simulations

---

# Directory Structure

```text
KV-Pipeline/
├── TIGRE/
│   └── Python/
│       ├── environment.yml                  # Conda environment dependencies
│       ├── setup.py                         # TIGRE compilation script
│       └── tigre/
│           └── utilities/
│               └── working_geometry.py      # Custom scanner physical geometry
│
├── SRC/
│   ├── config_raw_data.cfg                  # Pipeline config for physical scanner data
│   ├── config_simulated.cfg                 # Pipeline config for virtual simulations
│   ├── data_processing.py                   # Log projection and encoder alignment
│   ├── main.py                              # Main orchestrator for reconstruction
│   ├── pipeline_config.py                   # Safely parses .cfg files
│   ├── reconstruction.py                    # TIGRE algorithms (FDK, CGLS, etc.)
│   ├── simple_scene_creation.py             # Convert 3D .npy scenes from GUI into correct input
│   ├── visualization.py                     # 2D slicing and 3D PyVista rendering
│   ├── visualization_config_raw_data.cfg    # Visualization config for physical data
│   └── visualization_config_simulated.cfg   # Visualization config for simulations
│
├── Raw_Data/                                # Input directory for raw .dat scans
├── Raw_Data_Projections/                    # Output: processed projection arrays
├── Raw_Data_Recon/                          # Output: reconstructed 3D volumes
├── Simulated_Scene/                         # Input: synthetic 3D scenes
├── Simulated_Scene_Projections/             # Output: simulated projections
└── Simulated_Scene_Recon/                   # Output: reconstructed simulated volumes
```

---

# Key Components

## TIGRE/Python/

Contains the customized TIGRE installation and CUDA compilation setup.

| File | Description |
|---|---|
| `environment.yml` | Conda environment dependencies |
| `setup.py` | TIGRE CUDA/C++ compilation script |
| `working_geometry.py` | Custom scanner geometry definition |

---

## SRC/

Core reconstruction and visualization modules.

| File | Description |
|---|---|
| `config_raw_data.cfg` | Reconstruction config for physical scanner data |
| `config_simulated.cfg` | Reconstruction config for simulations |
| `data_processing.py` | Projection preprocessing and encoder synchronization |
| `main.py` | Main pipeline orchestrator |
| `pipeline_config.py` | Safe `.cfg` parser |
| `reconstruction.py` | Reconstruction algorithms (FDK, CGLS, etc.) |
| `simple_scene_creation.py` | Generates synthetic 3D scenes |
| `visualization.py` | Slice extraction and 3D rendering |
| `visualization_config_raw_data.cfg` | Visualization config for physical data |
| `visualization_config_simulated.cfg` | Visualization config for simulations |

---

# System Requirements

Because the pipeline relies on GPU-accelerated CUDA operations through TIGRE, your system must satisfy the following requirements.

## Operating System

- Windows 10 / 11
- Linux (Ubuntu 16.04+ recommended)

## Python

- Python 3.7 – 3.11
- Recommended: Python 3.10.13

## GPU

- NVIDIA CUDA-capable GPU
- Minimum Compute Capability: `>= 3.5`

## CUDA Toolkit

- CUDA 9.2 or newer
- CUDA 11.x or 12.x recommended

## C++ Compiler

### Windows

- MS Visual Studio Build Tools
- MSVC 19.24+ recommended
- Windows SDK required

### Linux

- `gcc >= 7.6.0`

---

# Setup Instructions

## 1. Install Conda

Install either Miniconda or Anaconda:

- https://docs.conda.io/en/latest/miniconda.html
- https://www.anaconda.com/download

---

## 2. Create the Conda Environment

Navigate to the TIGRE Python directory:

```bash
cd KV-Pipeline/TIGRE/Python/
```

Create the environment:

```bash
conda env create -f environment.yml
```

---

## 3. Compile and Install TIGRE

Activate the environment:

```bash
conda activate kv-env
```

Compile and install TIGRE:

```bash
python setup.py install
```

---

# Developer Notes

If you modify:

- CUDA/C++ files inside `TIGRE/`
- Python wrappers inside `tigre/`

you must rebuild TIGRE.

## Rebuild Commands

```bash
pip uninstall pytigre -y
python setup.py install
```

---

# Running the Reconstruction Pipeline

## Step 1 — Activate Environment

```bash
conda activate kv-env
```

---

## Step 2 — Configure the Pipeline

Review either:

- `config_raw_data.cfg`
- `config_simulated.cfg`

Verify:

- Input paths
- Air scan paths
- Geometry parameters
- Reconstruction algorithm selection

Supported algorithms include:

- FDK
- CGLS
- Other TIGRE iterative methods

---

## Step 3 — Execute the Pipeline

### Physical Scanner Reconstruction

```bash
cd ../../SRC/
python main.py --config config_raw_data.cfg
```

### Simulation Reconstruction

```bash
python main.py --config config_simulated.cfg
```

> **Important Note:**  
> If the simulated scene is generated using the GUI-based scene creator, you must first:
>
> 1. Modify the required input/output paths inside "simple_scene_creation.py"
>
> 2. Generate the synthetic 3D scene before running the reconstruction pipeline:
>
> ```bash
> python simple_scene_creation.py
> ```
>
> After the simulated `.npy` scene is generated, run the reconstruction pipeline using `config_simulated.cfg`.

---

# Reconstruction Outputs

The pipeline generates:

| Directory | Description |
|---|---|
| `Raw_Data_Projections/` | Processed projection arrays |
| `Raw_Data_Recon/` | Reconstructed physical volumes |
| `Simulated_Scene_Projections/` | Simulated projections |
| `Simulated_Scene_Recon/` | Simulated reconstructed volumes |

Generated outputs include:

- Projection GIFs
- Intermediate arrays
- Final reconstructed volume (`final_volume.npy`)

---

# Visualization Pipeline

After reconstruction, use the visualization module to extract 2D slices and optionally launch interactive 3D rendering.

## Step 1 — Configure Visualization

Review:

- `visualization_config_raw_data.cfg`
- `visualization_config_simulated.cfg`

Set:

- `volume_path`
- Slice indices
- Rendering options
- `show_3d = True` (optional)

---

## Step 2 — Execute Visualization

### Physical Data Visualization

```bash
python visualization.py --config visualization_config_raw_data.cfg
```

### Simulated Data Visualization

```bash
python visualization.py --config visualization_config_simulated.cfg
```

---

# Visualization Outputs

Generated outputs include:

| Output | Description |
|---|---|
| `.png` slice images | X/Y/Z slice visualizations |
| `slices/` directory | Saved slice images |
| PyVista window | Interactive 3D rendering |

---

# Features

- Modular CT reconstruction pipeline
- Physical scanner support
- Simulation pipeline support
- Config-driven workflow
- GPU-accelerated TIGRE backend
- Projection preprocessing
- Multiple reconstruction algorithms
- Interactive 3D visualization
- Slice extraction and export
- Flexible geometry customization

---

# Typical Workflow

```text
1. Prepare configuration file
2. Run reconstruction pipeline
3. Generate reconstructed volume
4. Configure visualization
5. Extract slices / render 3D scene
```