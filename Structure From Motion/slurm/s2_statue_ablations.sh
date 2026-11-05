#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=s2_statue_ablations
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

IMG="$CV/raw_images"

run --scene abl_full   --images "$IMG" $FROZEN
run --scene abl_noroot --images "$IMG" $FROZEN $SPARSE_ONLY --no-root-sift
run --scene abl_seq5   --images "$IMG" $FROZEN $SPARSE_ONLY --sequential-window 5
run --scene abl_akaze  --images "$IMG" $FROZEN $SPARSE_ONLY --feature akaze
run --scene abl_orb    --images "$IMG" $FROZEN $SPARSE_ONLY --feature orb
run --scene abl_nomm   --images "$IMG" $BASE   $SPARSE_ONLY
