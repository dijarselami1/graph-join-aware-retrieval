import json
import sqlite3
import re
from pathlib import Path
from itertools import combinations
from difflib import SequenceMatcher

ROOT = Path("/scratch/ds7491/graph_project")

SPIDER_DIR = ROOT / "data" / "spider"
DB_DIR = SPIDER_DIR / "databases"

META_DIR = ROOT / "metadata" / "spider"
OUTPUT_DIR = ROOT / "features" / "spider"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

with open(META_DIR / "columns.json") as f:
    columns = json.load(f)

with open(META_DIR / "foreign_keys.json") as f:
    foreign_keys = json.load(f)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_name(name):
    """
    Normalize a column name.

    Stadium_ID -> stadium id
    singerId   -> singer id
    """
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = " ".join(name.lower().split())
    return name


def name_similarity(a, b):
    a = normalize_name(a)
    b = normalize_name(b)

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def normalize_value(value):
    """
    Normalize values so that:

    1
    "1"

    can be compared.
    """
    if value is None:
        return None

    return str(value).strip().lower()


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def column_key(db_id, table, column):
    return (
        db_id.casefold(),
        table.casefold(),
        column.casefold()
    )


def pair_key(a, b):
    return tuple(sorted([a, b]))


# --------------------------------------------------
# Build GOLD join lookup
# --------------------------------------------------

gold_pairs = set()

for fk in foreign_keys:

    a = column_key(
        fk["db_id"],
        fk["from_table"],
        fk["from_column"]
    )

    b = column_key(
        fk["db_id"],
        fk["to_table"],
        fk["to_column"]
    )

    gold_pairs.add(
        pair_key(a, b)
    )


# --------------------------------------------------
# Organize column metadata by database
# --------------------------------------------------

columns_by_db = {}

for col in columns:

    columns_by_db.setdefault(
        col["db_id"],
        []
    ).append(col)


# --------------------------------------------------
# Output file
# --------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "candidate_column_pairs.jsonl"
)

out = open(output_file, "w")


total_pairs = 0
positive_pairs = 0
negative_pairs = 0


# --------------------------------------------------
# Process each database independently
# --------------------------------------------------

for db_index, (db_id, db_columns) in enumerate(
    sorted(columns_by_db.items()),
    1
):

    print(
        f"[{db_index}/{len(columns_by_db)}] "
        f"{db_id}"
    )

    sqlite_files = list(
        DB_DIR.rglob(f"{db_id}.sqlite")
    )

    if not sqlite_files:

        print(
            "  WARNING: SQLite database not found"
        )
        continue

    sqlite_path = sqlite_files[0]

    conn = sqlite3.connect(sqlite_path)
    conn.text_factory = lambda b: b.decode(
        "utf-8",
        errors="replace"
    )

    cur = conn.cursor()


    # ----------------------------------------------
    # Cache distinct values for every column
    # ----------------------------------------------

    value_cache = {}

    for col in db_columns:

        table = col["table"]
        column = col["column"]

        key = (
            table.casefold(),
            column.casefold()
        )

        try:

            cur.execute(
                f"""
                SELECT DISTINCT {quote(column)}
                FROM {quote(table)}
                WHERE {quote(column)} IS NOT NULL
                """
            )

            values = {
                normalize_value(row[0])
                for row in cur.fetchall()
            }

            values.discard(None)

            value_cache[key] = values

        except Exception as e:

            print(
                f"  WARNING values failed: "
                f"{table}.{column}: {e}"
            )

            value_cache[key] = set()


    # ----------------------------------------------
    # Compare columns from DIFFERENT tables
    # ----------------------------------------------

    for col_a, col_b in combinations(
        db_columns,
        2
    ):

        # We don't care about pairs inside
        # the same table for join discovery.
        if (
            col_a["table"].casefold()
            ==
            col_b["table"].casefold()
        ):
            continue


        values_a = value_cache.get(
            (
                col_a["table"].casefold(),
                col_a["column"].casefold()
            ),
            set()
        )

        values_b = value_cache.get(
            (
                col_b["table"].casefold(),
                col_b["column"].casefold()
            ),
            set()
        )


        # ------------------------------------------
        # Value-based features
        # ------------------------------------------

        intersection = values_a & values_b
        union = values_a | values_b

        intersection_size = len(intersection)

        jaccard = (
            intersection_size / len(union)
            if union
            else 0.0
        )

        a_in_b = (
            intersection_size / len(values_a)
            if values_a
            else 0.0
        )

        b_in_a = (
            intersection_size / len(values_b)
            if values_b
            else 0.0
        )

        min_overlap = (
            intersection_size /
            min(
                len(values_a),
                len(values_b)
            )
            if values_a and values_b
            else 0.0
        )


        # ------------------------------------------
        # Name features
        # ------------------------------------------

        normalized_name_a = normalize_name(
            col_a["column"]
        )

        normalized_name_b = normalize_name(
            col_b["column"]
        )

        exact_name_match = (
            normalized_name_a
            ==
            normalized_name_b
        )

        name_sim = name_similarity(
            col_a["column"],
            col_b["column"]
        )


        # ------------------------------------------
        # Type features
        # ------------------------------------------

        dtype_a = (
            str(col_a["datatype"])
            .strip()
            .lower()
        )

        dtype_b = (
            str(col_b["datatype"])
            .strip()
            .lower()
        )

        exact_type_match = (
            dtype_a == dtype_b
        )


        # ------------------------------------------
        # Gold label
        # ------------------------------------------

        key_a = column_key(
            db_id,
            col_a["table"],
            col_a["column"]
        )

        key_b = column_key(
            db_id,
            col_b["table"],
            col_b["column"]
        )

        is_gold_join = int(
            pair_key(key_a, key_b)
            in gold_pairs
        )


        # ------------------------------------------
        # Store record
        # ------------------------------------------

        record = {

            "db_id": db_id,

            "table_a": col_a["table"],
            "column_a": col_a["column"],

            "table_b": col_b["table"],
            "column_b": col_b["column"],

            "gold_join": is_gold_join,


            # names
            "name_similarity": round(
                name_sim,
                6
            ),

            "exact_name_match":
                exact_name_match,


            # values
            "distinct_a": len(values_a),
            "distinct_b": len(values_b),

            "intersection_size":
                intersection_size,

            "jaccard": round(
                jaccard,
                6
            ),

            "containment_a_in_b": round(
                a_in_b,
                6
            ),

            "containment_b_in_a": round(
                b_in_a,
                6
            ),

            "min_overlap": round(
                min_overlap,
                6
            ),


            # uniqueness
            "uniqueness_a":
                col_a[
                    "uniqueness_ratio_rows"
                ],

            "uniqueness_b":
                col_b[
                    "uniqueness_ratio_rows"
                ],


            # key information
            "pk_a":
                col_a["is_primary_key"],

            "pk_b":
                col_b["is_primary_key"],


            # datatypes
            "datatype_a":
                col_a["datatype"],

            "datatype_b":
                col_b["datatype"],

            "exact_type_match":
                exact_type_match,


            # row counts
            "row_count_a":
                col_a["row_count"],

            "row_count_b":
                col_b["row_count"]
        }


        out.write(
            json.dumps(record)
            + "\n"
        )


        total_pairs += 1

        if is_gold_join:
            positive_pairs += 1
        else:
            negative_pairs += 1


    conn.close()


out.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nCANDIDATE FEATURE DATASET")
print("=" * 50)

print("Total column pairs:", total_pairs)
print("Gold joins:", positive_pairs)
print("Non-gold pairs:", negative_pairs)

print("\nSaved to:")
print(output_file)
