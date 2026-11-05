#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=lens_statue_tele
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

run --scene lens_IMG_0369 --video "$CV/IMG_0369.mov" --video-fps 3 $FROZEN
