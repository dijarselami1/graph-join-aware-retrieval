import os
import csv
import pickle
import numpy as np

ROOT = "/scratch/ds7491/graph_project"
THRESHOLD = 0.5160912733321336

pairs = np.load(
    f"{ROOT}/features/spider/all_pairs/pair_indices.npy",
    mmap_mode="r"
)

scores = np.load(
    f"{ROOT}/features/spider/all_pairs/joinability_scores.npy",
    mmap_mode="r"
)

with open(f"{ROOT}/cache/spider/column_cache.pkl", "rb") as f:
    cache = pickle.load(f)

# Handle either a list or {"columns": [...]}
if isinstance(cache, dict) and "columns" in cache:
    columns = cache["columns"]
else:
    columns = cache

keep = np.flatnonzero(scores >= THRESHOLD)

os.makedirs(f"{ROOT}/graphs/spider", exist_ok=True)

column_edges = []
table_edges = {}

for pair_idx in keep:
    i, j = pairs[pair_idx]
    i, j = int(i), int(j)

    a = columns[i]
    b = columns[j]

    score = float(scores[pair_idx])

    db_a = a["db_id"]
    db_b = b["db_id"]
    table_a = a["table"]
    table_b = b["table"]
    col_a = a["column"]
    col_b = b["column"]

    column_edges.append([
        db_a, table_a, col_a,
        db_b, table_b, col_b,
        score
    ])

    # Table graph only needs relationships between different tables
    node_a = f"{db_a}.{table_a}"
    node_b = f"{db_b}.{table_b}"

    if node_a == node_b:
        continue

    # Canonical ordering
    if node_a > node_b:
        node_a, node_b = node_b, node_a
        db_a, db_b = db_b, db_a
        table_a, table_b = table_b, table_a
        col_a, col_b = col_b, col_a

    key = (node_a, node_b)

    # If multiple columns connect the same tables,
    # keep the strongest predicted join
    if key not in table_edges or score > table_edges[key][-1]:
        table_edges[key] = [
            db_a, table_a, col_a,
            db_b, table_b, col_b,
            score
        ]

column_edges.sort(key=lambda x: x[-1], reverse=True)
table_edge_rows = sorted(
    table_edges.values(),
    key=lambda x: x[-1],
    reverse=True
)

with open(f"{ROOT}/graphs/spider/predicted_column_edges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "db_a", "table_a", "column_a",
        "db_b", "table_b", "column_b",
        "score"
    ])
    writer.writerows(column_edges)

with open(f"{ROOT}/graphs/spider/predicted_table_edges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "db_a", "table_a", "column_a",
        "db_b", "table_b", "column_b",
        "score"
    ])
    writer.writerows(table_edge_rows)

print("Threshold:", THRESHOLD)
print("Predicted column edges:", len(column_edges))
print("Predicted table edges:", len(table_edge_rows))

print("\nTop 10 table edges:")
for row in table_edge_rows[:10]:
    print(row)
