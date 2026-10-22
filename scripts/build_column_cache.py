import json
import pickle
import sqlite3
from pathlib import Path

ROOT = Path("/scratch/ds7491/graph_project")

COLUMNS_FILE = ROOT / "metadata/spider/columns.json"
DB_ROOT = ROOT / "data/spider/databases"

OUT_DIR = ROOT / "cache/spider"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "column_cache.pkl"


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def normalize(v):
    if v is None:
        return None

    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")

    return str(v).strip().casefold()


with open(COLUMNS_FILE) as f:
    columns = json.load(f)

print("Columns:", len(columns))

cache = []

for i, c in enumerate(columns):

    db_id = c["db_id"]
    table = c["table"]
    column = c["column"]

    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

    values = set()

    try:
        conn = sqlite3.connect(str(db_path))

        # Avoid crashing on malformed UTF-8 values
        conn.text_factory = lambda b: b.decode(
            "utf-8",
            errors="replace"
        )

        query = (
            f"SELECT DISTINCT {quote(column)} "
            f"FROM {quote(table)} "
            f"WHERE {quote(column)} IS NOT NULL"
        )

        rows = conn.execute(query).fetchall()
        conn.close()

        for row in rows:
            value = normalize(row[0])

            if value not in (None, ""):
                values.add(value)

    except Exception as e:
        print(
            f"WARNING: {db_id}.{table}.{column}: {e}"
        )

    cache.append({
        "index": i,
        "db_id": db_id,
        "table": table,
        "column": column,
        "datatype": c["datatype"],

        "is_primary_key": bool(
            c["is_primary_key"]
        ),

        "row_count": c["row_count"],
        "non_null_count": c["non_null_count"],
        "distinct_count": len(values),

        "uniqueness_ratio": (
            c["uniqueness_ratio_non_null"]
        ),

        "values": values
    })

    if (i + 1) % 500 == 0:
        print(
            f"Cached {i + 1}/{len(columns)} columns"
        )


with open(OUT_FILE, "wb") as f:
    pickle.dump(
        cache,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\nDONE")
print("Cached columns:", len(cache))
print("Saved to:", OUT_FILE)
