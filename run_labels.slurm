#!/bin/bash
#SBATCH --job-name=pair_labels
#SBATCH --output=pair_labels.out
#SBATCH --error=pair_labels.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/build_pair_labels.py
