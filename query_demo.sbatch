#!/bin/bash
#SBATCH --job-name=query_paths
#SBATCH --output=logs/query_paths_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -e

cd /scratch/ds7491/graph_project
source .venv/bin/activate

python scripts/query_driven_paths.py \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer \
  --seed-b concert_singer.stadium \
  --edges graphs/spider/predicted_table_edges.csv \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --ranking graph --output results/query_graph_baseline.json

python scripts/query_driven_paths.py \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer \
  --seed-b concert_singer.stadium \
  --edges graphs/spider/predicted_table_edges.csv \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --ranking query-aware --query-weight 0.5 --length-penalty 0 \
  --output results/query_aware_paths.json
