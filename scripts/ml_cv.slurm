#!/bin/bash
#SBATCH --job-name=ml_cv
#SBATCH --output=ml_cv_%j.out
#SBATCH --error=ml_cv_%j.err
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
OUT="$DIR/results/ml_cv_results_%j.json"

mkdir -p "$DIR/results"

echo "[$(date)] Using Python: $(which python)"
python --version
echo "[$(date)] SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

srun python "$DIR/scripts/cv_feature.py" \
    --data "$DATA" \
    --out "$OUT" \
    --cv 10

echo "[$(date)] Done -> $OUT"
