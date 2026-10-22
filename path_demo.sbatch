#!/bin/bash
#SBATCH --job-name=graph_path
#SBATCH --output=logs/graph_path_%j.out
#SBATCH --time=00:05:00
#SBATCH --mem=2G

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/find_graph_path.py \
    concert_singer.singer \
    concert_singer.stadium
