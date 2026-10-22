import json
from pathlib import Path

ROOT = Path("/scratch/ds7491/graph_project")
META = ROOT / "metadata" / "spider"

with open(META / "columns.json") as f:
    columns = json.load(f)

with open(META / "foreign_keys.json") as f:
    foreign_keys = json.load(f)


# ---------------------------------------
# Existing columns, case-insensitive
# ---------------------------------------

column_lookup = {}

for c in columns:
    key = (
        c["db_id"].casefold(),
        c["table"].casefold(),
        c["column"].casefold()
    )

    column_lookup[key] = {
        "db_id": c["db_id"],
        "table": c["table"],
        "column": c["column"]
    }


resolved_pairs = set()
cross_table_pairs = set()
same_table_pairs = set()
unresolved = []


for fk in foreign_keys:

    source_key = (
        fk["db_id"].casefold(),
        fk["from_table"].casefold(),
        fk["from_column"].casefold()
    )

    target_key = (
        fk["db_id"].casefold(),
        fk["to_table"].casefold(),
        fk["to_column"].casefold()
    )

    source = column_lookup.get(source_key)
    target = column_lookup.get(target_key)

    if source is None or target is None:
        unresolved.append(fk)
        continue

    source_id = (
        source["db_id"].casefold(),
        source["table"].casefold(),
        source["column"].casefold()
    )

    target_id = (
        target["db_id"].casefold(),
        target["table"].casefold(),
        target["column"].casefold()
    )

    pair = tuple(sorted([source_id, target_id]))

    resolved_pairs.add(pair)

    if source["table"].casefold() == target["table"].casefold():
        same_table_pairs.add(pair)
    else:
        cross_table_pairs.add(pair)


print("\nGOLD JOIN COUNT CHECK")
print("=" * 50)

print("Original FK records:", len(foreign_keys))
print("Unique resolved FK pairs:", len(resolved_pairs))
print("Cross-table FK pairs:", len(cross_table_pairs))
print("Same-table FK pairs:", len(same_table_pairs))
print("Unresolved FK records:", len(unresolved))


print("\nSAME-TABLE FOREIGN KEYS")
print("=" * 50)

for pair in sorted(same_table_pairs):
    a, b = pair

    print(
        f"{a[0]}: "
        f"{a[1]}.{a[2]} "
        f"<-> "
        f"{b[1]}.{b[2]}"
    )


print("\nUNRESOLVED FOREIGN KEYS")
print("=" * 50)

for fk in unresolved:
    print(
        f"{fk['db_id']}: "
        f"{fk['from_table']}.{fk['from_column']} "
        f"-> "
        f"{fk['to_table']}.{fk['to_column']}"
    )
