#!/bin/bash
#SBATCH --job-name=elastic_cv_lofo
#SBATCH --output=elastic_cv_lofo%j.out
#SBATCH --error=elastic_cv_lofo%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail
module load apptainer/1.1.9

# Thread sanity (BLAS/OpenMP)
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_BLAS_NUM_THREADS=1
export SINGULARITYENV_R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.2"

# Paths
DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
IN="$DIR/data/CLR_micro.csv"
OUT_BASE="$DIR/results/elastic_net_${SLURM_JOB_ID}"
SCRIPT="$DIR/scripts/LOFO_en.R"
IMG="$HOME/Dermot_analysis/tidyverse_4.2.sif"   # your R image

# Params
K_OUTER=10
K_INNER=5
LAMBDA_CHOICE="lambda.1se"    # or lambda.min
SEED=2025
ALPHAS="0,0.25,0.5,0.75,1"
FLOCK_COL="breeder"

BIND_OPTS="--bind $HOME --bind /data/Genetics"
mkdir -p "$DIR/results"

echo "[$(date)] IMG: $IMG"
echo "[$(date)] SCRIPT: $SCRIPT"
echo "[$(date)] IN: $IN"
echo "[$(date)] OUT_BASE: $OUT_BASE"
echo "[$(date)] K_OUTER=$K_OUTER K_INNER=$K_INNER LAMBDA=$LAMBDA_CHOICE SEED=$SEED ALPHAS=$ALPHAS"

srun apptainer exec $BIND_OPTS "$IMG" \
  Rscript "$SCRIPT" "$IN" "$OUT_BASE" "$K_INNER" "$LAMBDA_CHOICE" "$SEED" "$FLOCK_COL" "$ALPHAS"

echo "[$(date)] Done. Outputs in $DIR/results/"
