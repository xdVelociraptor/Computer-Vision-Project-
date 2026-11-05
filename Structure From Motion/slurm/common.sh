# Sourced by every job. Submit from the project root: sbatch slurm/<job>.sh
# Order (2 at a time):
#   s1_light_robot          + lens_statue_tele
#   s2_statue_ablations     + lens_building_ultrawide
#   s3_dense_ablation       + lens_building_tele
#   lens_ultrawide_true_focal
# Statue wide/ultra-wide and building wide reuse the earlier fps3 runs
# (scene_01_fps3, scene_03_fps3, scene_02_fps3), made with these same settings.

set -eo pipefail

module load anaconda
set +u  # conda's activate hooks break under nounset
eval "$(conda shell.bash hook)"
conda activate sfm
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

cd "$SLURM_SUBMIT_DIR"
CV="${CV:-$SLURM_SUBMIT_DIR/CV}"

# Frozen settings shared by every run (dense thresholds = the pipeline defaults).
MIN_NCC=0.35
CONSISTENT_VIEWS=2

BASE="--feature-size 2400 --min-ncc $MIN_NCC --consistent-views $CONSISTENT_VIEWS"
FROZEN="$BASE --multi-model"
SPARSE_ONLY="--no-dense --no-surface"

run() {
    echo ">>> $(date '+%F %T') $*"
    python scripts/run_scratch_sfm.py "$@" || echo "FAILED: $*"
}
