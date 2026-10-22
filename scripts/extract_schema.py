import sqlite3
import json
import sys
from pathlib import Path

db_path = sys.argv[1]
output_path = sys.argv[2]

db_id = Path(db_path).stem

conn = sqlite3.connect(db_path)
cur = conn.cursor()

metadata = {
    "db_id": db_id,
    "database_path": db_path,
    "tables": {}
}

tables = cur.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
AND name NOT LIKE 'sqlite_%'
ORDER BY name
""").fetchall()

for (table,) in tables:

    columns = []

    for row in cur.execute(f'PRAGMA table_info("{table}")'):
        cid, name, dtype, notnull, default, pk = row

        columns.append({
            "name": name,
            "type": dtype,
            "not_null": bool(notnull),
            "primary_key": bool(pk)
        })

    foreign_keys = []

    for fk in cur.execute(f'PRAGMA foreign_key_list("{table}")'):
        foreign_keys.append({
            "from_column": fk[3],
            "to_table": fk[2],
            "to_column": fk[4]
        })

    row_count = cur.execute(
        f'SELECT COUNT(*) FROM "{table}"'
    ).fetchone()[0]

    metadata["tables"][table] = {
        "row_count": row_count,
        "columns": columns,
        "foreign_keys": foreign_keys
    }

conn.close()

with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to {output_path}")


