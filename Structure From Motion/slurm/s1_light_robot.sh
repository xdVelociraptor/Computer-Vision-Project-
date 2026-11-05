#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=s1_light_robot
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

run --scene fail_robot  --video "$CV/SecurityRobotData.MOV"      --video-fps 1 $FROZEN
run --scene light_night --video "$CV/waterFountainNight.mp4"     --video-fps 1 $FROZEN
run --scene light_day   --video "$CV/waterFountainAfternoon.mp4" --video-fps 1 $FROZEN
