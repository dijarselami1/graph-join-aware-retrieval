import json
import random
from pathlib import Path

ROOT = Path("/scratch/ds7491/graph_project")

INPUT_FILE = (
    ROOT / "features/spider/candidate_column_pairs.jsonl"
)

OUTPUT_DIR = (
    ROOT / "features/spider/splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load rows
# --------------------------------------------------

rows = []

with open(INPUT_FILE) as f:
    for line in f:
        rows.append(json.loads(line))

db_ids = sorted(set(r["db_id"] for r in rows))

print("Total databases:", len(db_ids))

# --------------------------------------------------
# Deterministic random split
# --------------------------------------------------

rng = random.Random(42)
rng.shuffle(db_ids)

train_size = int(len(db_ids) * 0.8)

train_dbs = set(db_ids[:train_size])
test_dbs = set(db_ids[train_size:])

# --------------------------------------------------
# Split rows by DATABASE
# --------------------------------------------------

train_rows = [
    r for r in rows
    if r["db_id"] in train_dbs
]

test_rows = [
    r for r in rows
    if r["db_id"] in test_dbs
]

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_DIR / "train_pairs.jsonl", "w") as f:
    for r in train_rows:
        f.write(json.dumps(r) + "\n")

with open(OUTPUT_DIR / "test_pairs.jsonl", "w") as f:
    for r in test_rows:
        f.write(json.dumps(r) + "\n")

with open(OUTPUT_DIR / "database_split.json", "w") as f:
    json.dump({
        "seed": 42,
        "train_databases": sorted(train_dbs),
        "test_databases": sorted(test_dbs)
    }, f, indent=2)

# --------------------------------------------------
# Statistics
# --------------------------------------------------

def stats(name, subset):
    positives = sum(r["gold_join"] == 1 for r in subset)
    negatives = len(subset) - positives

    print(f"\n{name}")
    print("-" * 40)
    print("Candidate pairs:", len(subset))
    print("Gold joins:", positives)
    print("Non-gold pairs:", negatives)

print("\nDATABASE SPLIT")
print("=" * 50)

print("Train databases:", len(train_dbs))
print("Test databases:", len(test_dbs))

print(
    "Database overlap:",
    len(train_dbs & test_dbs)
)

stats("TRAIN", train_rows)
stats("TEST", test_rows)

print("\nSaved to:")
print(OUTPUT_DIR)
