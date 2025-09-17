#!/bin/bash
#SBATCH --job-name=lmm_lofo
#SBATCH --output=lmm_lofo_%j.out
#SBATCH --error=lmm_lofo_%j.err
#SBATCH --time=8:00:00          # adjust walltime if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       # sommer::mmer is single-threaded
#SBATCH --mem=12G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dermot.kelly@teagasc.ie

set -euo pipefail
module load apptainer/1.1.9

# prevent threaded BLAS libraries from hogging cores
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_BLAS_NUM_THREADS=1
export APPTAINERENV_R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.2"

DIR="$HOME/Dermot_analysis/Phd/Paper_2/microbiome_ml"
SCRIPT="$DIR/scripts/lmm_withinflock.R"
IMG="$HOME/Dermot_analysis/tidyverse_4.2.sif"

BIND_OPTS="--bind $HOME --bind /data/Genetics"
mkdir -p "$DIR/results"

srun apptainer exec $BIND_OPTS "$IMG" Rscript "$SCRIPT"
