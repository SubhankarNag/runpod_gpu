#!/bin/bash -x
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH -J bagCT
#SBATCH -t 36:00:00
#SBATCH -o logs/%j.out               # name of stdout output file(--output)
#SBATCH -e logs/%j.err               # name of stderr error file(--error)

conda activate kv-env


cd $SLURM_WORKDIR
pwd
cd codes/bag_CT/KV-Pipeline/SRC/ || exit




python -u multiple_cgls_sim.py



conda deactivate

