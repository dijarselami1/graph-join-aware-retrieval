#!/bin/bash
#SBATCH --job-name=all_pairs
#SBATCH --output=all_pairs.out
#SBATCH --error=all_pairs.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

cd /scratch/ds7491/graph_project
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/build_all_pair_features.py
