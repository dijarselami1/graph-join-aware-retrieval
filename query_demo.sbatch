#!/bin/bash
#SBATCH --job-name=query_paths
#SBATCH --output=logs/query_paths_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/query_driven_paths.py \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer \
  --seed-b concert_singer.stadium
