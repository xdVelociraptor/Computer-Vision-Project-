#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=s3_dense_ablation
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Needs S2 finished. --reuse-sparse overwrites in place, so the frozen result
# is backed up first and restored afterwards.
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

OUT="outputs/abl_full"
[[ -f "$OUT/scratch/cameras.json" ]] || { echo "Run S2 first" >&2; exit 2; }

ALT=$([[ "$CONSISTENT_VIEWS" == 4 ]] && echo 2 || echo 4)
[[ -d "$OUT/scratch_cv$CONSISTENT_VIEWS" ]] || cp -r "$OUT/scratch" "$OUT/scratch_cv$CONSISTENT_VIEWS"

run --scene abl_full --images "$CV/raw_images" --reuse-sparse $BASE --consistent-views "$ALT"

rm -rf "$OUT/scratch_cv$ALT"
mv "$OUT/scratch" "$OUT/scratch_cv$ALT"
cp -r "$OUT/scratch_cv$CONSISTENT_VIEWS" "$OUT/scratch"
