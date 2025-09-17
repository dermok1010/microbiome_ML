#!/bin/bash
#SBATCH --job-name=ml_cv_lofo
#SBATCH --output=ml_cv_lofo%j.out
#SBATCH --error=ml_cv_lofo%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail

# Activate your conda (same as your Jupyter script)
source /install/software/2024/minpy3/python3.10/etc/profile.d/conda.sh
conda activate /data/Genetics/analysis/R1681_OviSeq/Dermot/envs/dermo_python_env

# Keep math libs single-threaded; let sklearn/xgboost use cpus-per-task
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1

DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
DATA="$DIR/data/CLR_micro.csv"
OUT_PREFIX="$DIR/results/ml_lofo_${SLURM_JOB_ID}"

mkdir -p "$DIR/results"

echo "[$(date)] Using Python: $(which python)"
python --version
echo "[$(date)] SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Run LOFO CV
srun python "$DIR/scripts/ml_cv_lofo.py" \
    --data "$DATA" \
    --out_prefix "$OUT_PREFIX" \
    --breeder_col breeder \
    --inner_k 5 \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --seed 2025 \
    --models both

echo "[$(date)] Done -> ${OUT_PREFIX}_lofo_metrics.csv"
