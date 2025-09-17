#!/bin/bash
#SBATCH --job-name=lofo_pi
#SBATCH --output=lofo_pi_%j.out
#SBATCH --error=lofo_pi_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail

# Activate conda env
source /install/software/2024/minpy3/python3.10/etc/profile.d/conda.sh
conda activate /data/Genetics/analysis/R1681_OviSeq/Dermot/envs/dermo_python_env

# Keep math libs single-threaded; let sklearn/xgboost use cpus-per-task
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1

# Paths
DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
DATA="$DIR/data/CLR_micro.csv"
OUT_PREFIX="$DIR/results/lofo_pi_${SLURM_JOB_ID}"

mkdir -p "$DIR/results"

echo "[$(date)] Using Python: $(which python)"
python --version
echo "[$(date)] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# Run LOFO permutation-importance
srun python "$DIR/scripts/lofo_importances.py" \
  --data "$DATA" \
  --out_prefix "$OUT_PREFIX" \
  --breeder_col breeder \
  --genera_start 343 --genera_end 501 \
  --pi_repeats 30 \
  --topk 20 \
  --models both \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 2025

echo "[$(date)] Done."
echo "Outputs:"
echo "  ${OUT_PREFIX}_lobo_metrics.csv"
echo "  ${OUT_PREFIX}_rf_genus_importance_matrix.csv"
echo "  ${OUT_PREFIX}_xgb_genus_importance_matrix.csv"
echo "  ${OUT_PREFIX}_top20_genus_per_breeder.csv"
