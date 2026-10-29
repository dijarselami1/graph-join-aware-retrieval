#!/bin/bash
#SBATCH --job-name=download_minilm
#SBATCH --output=logs/download_minilm_%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -e
cd /scratch/ds7491/graph_project
module load python/3.11.3
source .venv-embeddings/bin/activate

python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)
model.save("/scratch/ds7491/graph_project/models/all-MiniLM-L6-v2")
print("MiniLM downloaded and saved successfully")
PY
