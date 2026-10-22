#!/bin/bash
#SBATCH --job-name=score_all
#SBATCH --output=score_all.out
#SBATCH --error=score_all.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/score_all_pairs.py
