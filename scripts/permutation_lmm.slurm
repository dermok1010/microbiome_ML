#!/bin/bash
#SBATCH --job-name=perm_lmm_ctr
#SBATCH --output=perm_lmm_ctr_%j.out
#SBATCH --error=perm_lmm_ctr_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail
module load apptainer/1.1.9

# Pass env vars into the container
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_BLAS_NUM_THREADS=1
export SINGULARITYENV_R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.2"

# 4) Working directory and params (absolute paths)
DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
N=10000
K=50
SEED=134
OUT="$DIR/results/random_var_explained_N${N}_k${K}_seed${SEED}.rds"
SCRIPT="$DIR/scripts/permute_priority_min.R"
IMG="$HOME/Dermot_analysis/tidyverse_4.2.sif"   # path to your image

# Bind your home (contains DIR + user lib); add extra binds if needed
BIND_OPTS="--bind $HOME --bind /data/Genetics"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

echo "[$(date)] IMG: $IMG"
echo "[$(date)] SCRIPT: $SCRIPT"
echo "[$(date)] OUT: $OUT"
echo "[$(date)] N=$N K=$K SEED=$SEED"

srun apptainer exec $BIND_OPTS "$IMG" \
     Rscript "$SCRIPT" "$N" "$K" "$SEED" "$OUT"

echo "[$(date)] Done -> $OUT"
