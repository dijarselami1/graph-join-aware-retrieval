import sqlite3
import json
import argparse
from pathlib import Path


SPIDER_DIR = Path("/scratch/ds7491/graph_project/data/spider")
DATABASE_DIR = SPIDER_DIR / "databases"
OUTPUT_DIR = Path("/scratch/ds7491/graph_project/metadata/spider")


def q(identifier):
    """Safely quote a SQLite table/column name."""
    return '"' + identifier.replace('"', '""') + '"'


def make_json_safe(value):
    """Make SQLite values safe to store in JSON."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def inspect_database(sqlite_path):
    db_id = sqlite_path.stem

    conn = sqlite3.connect(sqlite_path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cur = conn.cursor()

    # Find user tables
    cur.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    tables = [row[0] for row in cur.fetchall()]

    database_info = {
        "db_id": db_id,
        "sqlite_path": str(sqlite_path),
        "num_tables": len(tables),
        "tables": tables
    }

    column_records = []
    foreign_key_records = []

    for table in tables:

        # -------------------------
        # Row count
        # -------------------------
        cur.execute(f"SELECT COUNT(*) FROM {q(table)}")
        row_count = cur.fetchone()[0]

        # -------------------------
        # Column metadata
        # -------------------------
        cur.execute(f"PRAGMA table_info({q(table)})")
        columns = cur.fetchall()

        for col in columns:
            cid, column_name, dtype, notnull, default_value, pk_position = col

            # Number of non-null values
            cur.execute(
                f"SELECT COUNT({q(column_name)}) "
                f"FROM {q(table)}"
            )
            non_null_count = cur.fetchone()[0]

            # Number of distinct non-null values
            cur.execute(
                f"SELECT COUNT(DISTINCT {q(column_name)}) "
                f"FROM {q(table)} "
                f"WHERE {q(column_name)} IS NOT NULL"
            )
            distinct_count = cur.fetchone()[0]

            # Same uniqueness measure we manually tested before
            uniqueness_ratio_rows = (
                distinct_count / row_count
                if row_count > 0 else 0
            )

            # Also store uniqueness among only non-null values
            uniqueness_ratio_non_null = (
                distinct_count / non_null_count
                if non_null_count > 0 else 0
            )

            # Sample values
            cur.execute(
                f"SELECT DISTINCT {q(column_name)} "
                f"FROM {q(table)} "
                f"WHERE {q(column_name)} IS NOT NULL "
                f"LIMIT 5"
            )

            samples = [
                make_json_safe(row[0])
                for row in cur.fetchall()
            ]

            record = {
                "db_id": db_id,
                "table": table,
                "column": column_name,

                "datatype": dtype,

                "row_count": row_count,
                "non_null_count": non_null_count,
                "null_count": row_count - non_null_count,

                "distinct_count": distinct_count,

                "uniqueness_ratio_rows": round(
                    uniqueness_ratio_rows, 6
                ),

                "uniqueness_ratio_non_null": round(
                    uniqueness_ratio_non_null, 6
                ),

                "is_primary_key": pk_position > 0,

                # Important for composite PKs
                "primary_key_position": pk_position,

                "not_null": bool(notnull),

                "sample_values": samples
            }

            column_records.append(record)

        # -------------------------
        # Foreign keys
        # -------------------------
        cur.execute(f"PRAGMA foreign_key_list({q(table)})")

        for fk in cur.fetchall():
            (
                fk_id,
                sequence,
                referenced_table,
                from_column,
                to_column,
                on_update,
                on_delete,
                match
            ) = fk

            foreign_key_records.append({
                "db_id": db_id,

                "from_table": table,
                "from_column": from_column,

                "to_table": referenced_table,
                "to_column": to_column,

                "on_update": on_update,
                "on_delete": on_delete
            })

    conn.close()

    return database_info, column_records, foreign_key_records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=str,
        help="Only process one database, e.g. concert_singer"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N databases"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sqlite_files = sorted(DATABASE_DIR.rglob("*.sqlite"))

    if args.db:
        sqlite_files = [
            p for p in sqlite_files
            if p.stem == args.db
        ]

    if args.limit:
        sqlite_files = sqlite_files[:args.limit]

    print(f"Found {len(sqlite_files)} databases to process.")

    all_databases = []
    all_columns = []
    all_foreign_keys = []

    for i, sqlite_path in enumerate(sqlite_files, 1):

        print(
            f"[{i}/{len(sqlite_files)}] "
            f"Processing {sqlite_path.stem}"
        )

        try:
            db_info, columns, foreign_keys = inspect_database(
                sqlite_path
            )

            all_databases.append(db_info)
            all_columns.extend(columns)
            all_foreign_keys.extend(foreign_keys)

        except Exception as e:
            print(
                f"ERROR processing {sqlite_path}: {e}"
            )

    # -------------------------
    # Save outputs
    # -------------------------

    with open(
        OUTPUT_DIR / "databases.json",
        "w"
    ) as f:
        json.dump(all_databases, f, indent=2)

    with open(
        OUTPUT_DIR / "columns.json",
        "w"
    ) as f:
        json.dump(all_columns, f, indent=2)

    with open(
        OUTPUT_DIR / "foreign_keys.json",
        "w"
    ) as f:
        json.dump(all_foreign_keys, f, indent=2)

    print("\nDONE")
    print("---------------------------")
    print("Databases:", len(all_databases))
    print("Columns:", len(all_columns))
    print("Foreign keys:", len(all_foreign_keys))

    print("\nSaved to:")
    print(OUTPUT_DIR / "databases.json")
    print(OUTPUT_DIR / "columns.json")
    print(OUTPUT_DIR / "foreign_keys.json")


if __name__ == "__main__":
    main()
