#!/bin/bash
#SBATCH --job-name=query_compare
#SBATCH --output=logs/query_compare_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -eu
cd /scratch/ds7491/graph_project
# Use a separate embedding-compatible environment if the original venv is too old.
source "${EMBEDDING_VENV:-.venv}/bin/activate"
: "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL to a local SentenceTransformer model directory}"

# Offline graph/lexical/embedding comparison. This job never enables --llm.
python scripts/query_driven_paths.py \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --edges graphs/spider/predicted_table_edges.csv \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --compare --ranking embedding --embedding-model "$EMBEDDING_MODEL" \
  --query-weight 0.5 --embedding-weight 0.5 --length-penalty 0 \
  --output results/query_comparison.json
