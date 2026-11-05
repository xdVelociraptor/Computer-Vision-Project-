#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=lens_building_ultrawide
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

run --scene lens_IMG_0364 --video "$CV/IMG_0364.mov" --video-fps 3 $FROZEN
