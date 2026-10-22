#!/bin/bash
#SBATCH --job-name=join_rf
#SBATCH --output=join_rf.out
#SBATCH --error=join_rf.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/train_joinability_rf.py
