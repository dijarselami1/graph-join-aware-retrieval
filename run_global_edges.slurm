#!/bin/bash

#SBATCH --job-name=global_edges
#SBATCH --output=global_edges.out
#SBATCH --error=global_edges.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=users

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/build_global_structural_edges.py
