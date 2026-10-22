import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/ds7491/graph_project")

with open(ROOT / "cache/spider/column_cache.pkl", "rb") as f:
    cols = pickle.load(f)

with open(ROOT / "metadata/spider/foreign_keys.json") as f:
    fks = json.load(f)

pairs = np.load(
    ROOT / "features/spider/all_pairs/pair_indices.npy",
    mmap_mode="r"
)

n = len(cols)
num_pairs = len(pairs)

# -1 = unknown by default
labels = np.lib.format.open_memmap(
    ROOT / "features/spider/all_pairs/labels.npy",
    mode="w+",
    dtype=np.int8,
    shape=(num_pairs,)
)
labels[:] = -1

# Give each database an integer ID
db_names = [c["db_id"] for c in cols]
db_to_int = {db: i for i, db in enumerate(sorted(set(db_names)))}
db_ids = np.array([db_to_int[x] for x in db_names])

# Same-database pairs become negatives initially
chunk = 1000000

for start in range(0, num_pairs, chunk):
    end = min(start + chunk, num_pairs)

    a = pairs[start:end, 0]
    b = pairs[start:end, 1]

    same_db = db_ids[a] == db_ids[b]

    labels[start:end][same_db] = 0

# Map every column to its index
column_lookup = {}

for i, c in enumerate(cols):
    key = (
        c["db_id"].casefold(),
        c["table"].casefold(),
        c["column"].casefold()
    )
    column_lookup[key] = i

# Convert known FKs into positive labels
resolved = 0
unresolved = 0

for fk in fks:
    key_a = (
        fk["db_id"].casefold(),
        fk["from_table"].casefold(),
        fk["from_column"].casefold()
    )

    key_b = (
        fk["db_id"].casefold(),
        fk["to_table"].casefold(),
        fk["to_column"].casefold()
    )

    if key_a not in column_lookup or key_b not in column_lookup:
        unresolved += 1
        continue

    i = column_lookup[key_a]
    j = column_lookup[key_b]

    if i == j:
        continue

    if i > j:
        i, j = j, i

    # Position of pair (i,j) in pair_indices.npy
    pair_index = i * (2 * n - i - 1) // 2 + (j - i - 1)

    labels[pair_index] = 1
    resolved += 1

labels.flush()

print("Total pairs:", num_pairs)
print("Positive FK pairs:", np.sum(labels == 1))
print("Same-database negatives:", np.sum(labels == 0))
print("Cross-database unknown:", np.sum(labels == -1))
print("FK records resolved:", resolved)
print("FK records unresolved:", unresolved)
print("Saved labels.npy")
