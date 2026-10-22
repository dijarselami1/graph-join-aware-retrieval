#!/bin/bash
#SBATCH --job-name=column_cache
#SBATCH --output=column_cache.out
#SBATCH --error=column_cache.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/build_column_cache.py
