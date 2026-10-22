import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

ROOT = Path("/scratch/ds7491/graph_project")

with open(ROOT / "cache/spider/column_cache.pkl", "rb") as f:
    cols = pickle.load(f)

emb = np.load(ROOT / "embeddings/spider/column_embeddings.npy").astype(np.float32)

n = len(cols)
num_pairs = n * (n - 1) // 2
workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

OUT = ROOT / "features/spider/all_pairs"
OUT.mkdir(parents=True, exist_ok=True)

print("Columns:", n)
print("Pairs:", num_pairs)
print("Workers:", workers)


# --------------------------------------------------
# 1. Semantic cosine similarity
# --------------------------------------------------

print("Computing embedding similarities...")

norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb = emb / np.maximum(norms, 1e-12)

cosine_matrix = emb @ emb.T


# --------------------------------------------------
# 2. Name similarities
# --------------------------------------------------

print("Computing column-name similarities...")

column_names = [c["column"].lower() for c in cols]

column_name_matrix = process.cdist(
    column_names,
    column_names,
    scorer=fuzz.ratio,
    dtype=np.float32,
    workers=workers,
)
column_name_matrix *= 0.01


print("Computing table-name similarities...")

table_names = [c["table"].lower() for c in cols]

table_name_matrix = process.cdist(
    table_names,
    table_names,
    scorer=fuzz.ratio,
    dtype=np.float32,
    workers=workers,
)
table_name_matrix *= 0.01


# --------------------------------------------------
# 3. Value intersections
# --------------------------------------------------

print("Building value index...")

value_to_columns = defaultdict(list)

for i, c in enumerate(cols):
    for value in c["values"]:
        value_to_columns[value].append(i)

starts = np.array(
    [i * (2 * n - i - 1) // 2 for i in range(n)],
    dtype=np.int64
)

intersection_counts = np.zeros(num_pairs, dtype=np.uint32)

print("Computing value intersections...")

for k, column_list in enumerate(value_to_columns.values()):

    if len(column_list) < 2:
        continue

    column_list.sort()

    for p in range(len(column_list) - 1):

        i = column_list[p]

        js = np.asarray(
            column_list[p + 1:],
            dtype=np.int64
        )

        indices = starts[i] + (js - i - 1)

        np.add.at(
            intersection_counts,
            indices,
            1
        )

    if (k + 1) % 100000 == 0:
        print("Processed values:", k + 1)


# --------------------------------------------------
# 4. Prepare metadata arrays
# --------------------------------------------------

value_counts = np.array(
    [len(c["values"]) for c in cols],
    dtype=np.float32
)

pk = np.array(
    [float(c["is_primary_key"]) for c in cols],
    dtype=np.float32
)

uniqueness = np.array(
    [float(c["uniqueness_ratio"] or 0) for c in cols],
    dtype=np.float32
)


def type_family(dtype):
    t = (dtype or "").upper()

    if any(x in t for x in [
        "INT", "REAL", "FLOAT", "DOUBLE",
        "NUM", "DECIMAL"
    ]):
        return "numeric"

    if any(x in t for x in [
        "CHAR", "TEXT", "CLOB", "VARCHAR"
    ]):
        return "text"

    if "DATE" in t or "TIME" in t:
        return "datetime"

    if "BOOL" in t:
        return "boolean"

    return t or "unknown"


type_families = np.array([
    type_family(c["datatype"])
    for c in cols
])


# --------------------------------------------------
# 5. Create output arrays
# --------------------------------------------------

feature_names = [
    "embedding_cosine",
    "column_name_similarity",
    "table_name_similarity",
    "value_overlap",
    "jaccard",
    "pk_a",
    "pk_b",
    "uniqueness_a",
    "uniqueness_b",
    "datatype_compatible",
    "distinct_count_ratio",
]

features = np.lib.format.open_memmap(
    OUT / "features.npy",
    mode="w+",
    dtype=np.float32,
    shape=(num_pairs, len(feature_names)),
)

pairs = np.lib.format.open_memmap(
    OUT / "pair_indices.npy",
    mode="w+",
    dtype=np.int32,
    shape=(num_pairs, 2),
)


# --------------------------------------------------
# 6. Generate every pair
# --------------------------------------------------

print("Writing pair features...")

for i in range(n - 1):

    js = np.arange(i + 1, n)

    start = starts[i]
    end = start + len(js)

    intersections = intersection_counts[start:end].astype(np.float32)

    min_counts = np.minimum(
        value_counts[i],
        value_counts[js]
    )

    max_counts = np.maximum(
        value_counts[i],
        value_counts[js]
    )

    unions = (
        value_counts[i]
        + value_counts[js]
        - intersections
    )

    overlap = np.divide(
        intersections,
        min_counts,
        out=np.zeros_like(intersections),
        where=min_counts > 0,
    )

    jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )

    distinct_ratio = np.divide(
        min_counts,
        max_counts,
        out=np.zeros_like(min_counts),
        where=max_counts > 0,
    )

    datatype_compatible = (
        type_families[i] == type_families[js]
    ).astype(np.float32)

    pairs[start:end, 0] = i
    pairs[start:end, 1] = js

    features[start:end, 0] = cosine_matrix[i, js]
    features[start:end, 1] = column_name_matrix[i, js]
    features[start:end, 2] = table_name_matrix[i, js]
    features[start:end, 3] = overlap
    features[start:end, 4] = jaccard
    features[start:end, 5] = pk[i]
    features[start:end, 6] = pk[js]
    features[start:end, 7] = uniqueness[i]
    features[start:end, 8] = uniqueness[js]
    features[start:end, 9] = datatype_compatible
    features[start:end, 10] = distinct_ratio

    if (i + 1) % 500 == 0:
        print(f"Finished column {i + 1}/{n}")


features.flush()
pairs.flush()

with open(OUT / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(OUT / "manifest.json", "w") as f:
    json.dump({
        "columns": n,
        "pairs": num_pairs,
        "features": feature_names
    }, f, indent=2)

print("\nDONE")
print("Pairs:", num_pairs)
print("Feature shape:", features.shape)
print("Saved to:", OUT)
