#!/bin/bash
#SBATCH --job-name=edge_export
#SBATCH --output=logs/edge_export_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/export_predicted_edges.py
