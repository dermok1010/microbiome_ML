#!/bin/bash
#SBATCH --job-name=pcr_cov3pcs
#SBATCH --output=pcr_cov3pcs_%j.out
#SBATCH --error=pcr_cov3pcs_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail
module load apptainer/1.1.9

export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_BLAS_NUM_THREADS=1
export APPTAINERENV_R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.2"

DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
IN="$DIR/data/CLR_micro.csv"
OUT_BASE="$DIR/results/pcr_cov3pcs_${SLURM_JOB_ID}"
SCRIPT="$DIR/scripts/PC_prediction.R"
IMG="$HOME/Dermot_analysis/tidyverse_4.2.sif"

SEED=2025
FLOCK_COL="breeder"

BIND_OPTS="--bind $HOME --bind /data/Genetics"
mkdir -p "$DIR/results"

srun apptainer exec $BIND_OPTS "$IMG" \
  Rscript "$SCRIPT" "$IN" "$OUT_BASE" "$SEED" "$FLOCK_COL"
