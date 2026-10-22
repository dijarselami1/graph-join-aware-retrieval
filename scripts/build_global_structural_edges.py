import json
import sqlite3
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/scratch/ds7491/graph_project")

COLUMNS_FILE = ROOT / "metadata/spider/columns.json"
EMBEDDINGS_FILE = ROOT / "embeddings/spider/column_embeddings.npy"
DB_ROOT = ROOT / "data/spider/databases"

OUT_DIR = ROOT / "features/spider/global"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILE = OUT_DIR / "top50_structural_candidates.jsonl"
EDGE_FILE = OUT_DIR / "predicted_join_edges.jsonl"

K = 50


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_value(v):
    if v is None:
        return None
    return str(v).strip().casefold()


def name_similarity(a, b):
    return SequenceMatcher(
        None,
        a.casefold(),
        b.casefold()
    ).ratio()


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------
# Load metadata + embeddings
# --------------------------------------------------

with open(COLUMNS_FILE) as f:
    columns = json.load(f)

embeddings = np.load(EMBEDDINGS_FILE)

print("Columns:", len(columns))
print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# GLOBAL top-50 embedding retrieval
# --------------------------------------------------

nn = NearestNeighbors(
    n_neighbors=min(len(columns), K + 50),
    metric="cosine",
    algorithm="brute"
)

nn.fit(embeddings)

print("\nFinding global top-50 candidates...")

distances, indices = nn.kneighbors(embeddings)

candidate_pairs = set()

for i, neighbors in enumerate(indices):

    source = columns[i]
    kept = 0

    for j in neighbors:

        if i == j:
            continue

        target = columns[j]

        # Skip only the exact same physical table.
        # Different databases remain fully allowed.
        if (
            source["db_id"] == target["db_id"]
            and source["table"].casefold()
                == target["table"].casefold()
        ):
            continue

        candidate_pairs.add(
            tuple(sorted((i, j)))
        )

        kept += 1

        if kept >= K:
            break


print("Unique top-50 candidate pairs:", len(candidate_pairs))


# --------------------------------------------------
# Read real distinct values from SQLite
# Cache each column once
# --------------------------------------------------

value_cache = {}


def get_values(index):

    if index in value_cache:
        return value_cache[index]

    c = columns[index]

    db_id = c["db_id"]
    table = c["table"]
    column = c["column"]

    db_path = (
        DB_ROOT /
        db_id /
        f"{db_id}.sqlite"
    )

    values = set()

    try:
        conn = sqlite3.connect(str(db_path))

        query = (
            f"SELECT DISTINCT "
            f"{quote_identifier(column)} "
            f"FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} IS NOT NULL"
        )

        rows = conn.execute(query).fetchall()

        conn.close()

        for row in rows:
            v = normalize_value(row[0])

            if v not in (None, ""):
                values.add(v)

    except Exception as e:
        print(
            f"WARNING reading "
            f"{db_id}.{table}.{column}: {e}"
        )

    value_cache[index] = values
    return values


# --------------------------------------------------
# Structural feature calculation
# --------------------------------------------------

predicted_edges = 0
processed = 0

with open(CANDIDATE_FILE, "w") as candidates_out, \
     open(EDGE_FILE, "w") as edges_out:

    for i, j in sorted(candidate_pairs):

        a = columns[i]
        b = columns[j]

        values_a = get_values(i)
        values_b = get_values(j)

        intersection = values_a & values_b

        if values_a and values_b:
            min_overlap = (
                len(intersection) /
                min(len(values_a), len(values_b))
            )
        else:
            min_overlap = 0.0

        n_sim = name_similarity(
            a["column"],
            b["column"]
        )

        pk_evidence = (
            a["is_primary_key"]
            or b["is_primary_key"]
        )

        record = {
            "index_a": int(i),
            "index_b": int(j),

            "db_a": a["db_id"],
            "table_a": a["table"],
            "column_a": a["column"],

            "db_b": b["db_id"],
            "table_b": b["table"],
            "column_b": b["column"],

            "name_similarity": round(n_sim, 6),
            "min_overlap": round(min_overlap, 6),

            "pk_a": bool(a["is_primary_key"]),
            "pk_b": bool(b["is_primary_key"]),

            "distinct_a": len(values_a),
            "distinct_b": len(values_b),
            "intersection_size": len(intersection),

            "same_original_database": (
                a["db_id"] == b["db_id"]
            )
        }

        candidates_out.write(
            json.dumps(record) + "\n"
        )

        # Our frozen structural rule
        is_edge = (
            n_sim >= 0.85
            and min_overlap >= 0.85
            and pk_evidence
        )

        if is_edge:
            edges_out.write(
                json.dumps(record) + "\n"
            )
            predicted_edges += 1

        processed += 1

        if processed % 10000 == 0:
            print(
                f"Processed {processed}/{len(candidate_pairs)} "
                f"| predicted edges: {predicted_edges}"
            )


print("\nDONE")
print("Candidate pairs processed:", processed)
print("Predicted join edges:", predicted_edges)
print("Candidates saved to:", CANDIDATE_FILE)
print("Edges saved to:", EDGE_FILE)
