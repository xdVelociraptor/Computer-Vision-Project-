#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=lens_building_ultrawide_true_focal
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# 891 px = 14 mm equivalent at --feature-size 2400 (from the lens metadata).
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

run --scene lens_IMG_0364_f891 --video "$CV/IMG_0364.mov" --video-fps 3 $FROZEN --focal 891
