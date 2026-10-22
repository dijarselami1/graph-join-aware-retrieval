import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/scratch/ds7491/graph_project")

COLUMNS = ROOT / "metadata/spider/columns.json"
EMBEDDINGS = ROOT / "embeddings/spider/column_embeddings.npy"

TRAIN = ROOT / "features/spider/splits/train_pairs_embeddings.jsonl"
TEST = ROOT / "features/spider/splits/test_pairs_embeddings.jsonl"

K_VALUES = [5, 10, 20, 50, 100]


def key(db, table, column):
    return (
        db.casefold(),
        table.casefold(),
        column.casefold()
    )


# --------------------------------------------------
# Load columns + embeddings
# --------------------------------------------------

with open(COLUMNS) as f:
    columns = json.load(f)

embeddings = np.load(EMBEDDINGS)

column_keys = [
    key(c["db_id"], c["table"], c["column"])
    for c in columns
]

index_lookup = {
    k: i for i, k in enumerate(column_keys)
}

print("Columns:", len(columns))
print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# Load all known cross-table gold joins
# --------------------------------------------------

gold_pairs = set()

for path in [TRAIN, TEST]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)

            if r["gold_join"] != 1:
                continue

            a = key(
                r["db_id"],
                r["table_a"],
                r["column_a"]
            )

            b = key(
                r["db_id"],
                r["table_b"],
                r["column_b"]
            )

            pair = tuple(sorted([a, b]))
            gold_pairs.add(pair)

print("Known cross-table gold joins:", len(gold_pairs))


# --------------------------------------------------
# Global nearest-neighbor search
# NO db_id filtering
# --------------------------------------------------

MAX_K = max(K_VALUES)

# retrieve extra neighbors because some may belong
# to the same physical table and will be skipped
neighbors_to_fetch = min(len(columns), MAX_K + 50)

nn = NearestNeighbors(
    n_neighbors=neighbors_to_fetch,
    metric="cosine",
    algorithm="brute"
)

nn.fit(embeddings)

print("\nRunning GLOBAL nearest-neighbor search...")

distances, indices = nn.kneighbors(embeddings)


# --------------------------------------------------
# Build top-K candidate lists
# --------------------------------------------------

candidate_lists = {}

for i, neighbor_indices in enumerate(indices):

    source = columns[i]

    valid = []

    for j in neighbor_indices:

        if i == j:
            continue

        target = columns[j]

        # Skip columns from the exact same physical table.
        # We are looking for joins BETWEEN tables.
        if (
            source["db_id"] == target["db_id"]
            and source["table"].casefold() == target["table"].casefold()
        ):
            continue

        valid.append(j)

        if len(valid) >= MAX_K:
            break

    candidate_lists[i] = valid


# --------------------------------------------------
# Evaluate gold-pair recovery
# --------------------------------------------------

print("\nGLOBAL TOP-K GOLD JOIN RECOVERY")
print("=" * 55)

for K in K_VALUES:

    recovered = 0

    for a, b in gold_pairs:

        ia = index_lookup.get(a)
        ib = index_lookup.get(b)

        if ia is None or ib is None:
            continue

        # Pair counts as recovered if either endpoint
        # retrieves the other.
        found = (
            ib in candidate_lists[ia][:K]
            or ia in candidate_lists[ib][:K]
        )

        if found:
            recovered += 1

    recall = recovered / len(gold_pairs)

    # unique global candidate pairs
    candidate_pairs = set()

    for i, neighbors in candidate_lists.items():
        for j in neighbors[:K]:
            candidate_pairs.add(
                tuple(sorted((i, j)))
            )

    print(
        f"K={K:<3} "
        f"Gold recovered={recovered}/{len(gold_pairs)} "
        f"Recall={recall:.3f} "
        f"Candidate pairs={len(candidate_pairs)}"
    )

print("\nDONE")
