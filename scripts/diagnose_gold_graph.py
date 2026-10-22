import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/ds7491/graph_project")
META = ROOT / "metadata" / "spider"

with open(META / "columns.json") as f:
    columns = json.load(f)

with open(META / "foreign_keys.json") as f:
    fks = json.load(f)


def node_id(db, table, column):
    return f"{db}##{table}##{column}"


# Exact column IDs
exact_nodes = {
    node_id(
        c["db_id"],
        c["table"],
        c["column"]
    )
    for c in columns
}


# Case-insensitive lookup
casefold_lookup = defaultdict(list)

for c in columns:
    key = (
        c["db_id"].casefold(),
        c["table"].casefold(),
        c["column"].casefold()
    )

    casefold_lookup[key].append(
        node_id(
            c["db_id"],
            c["table"],
            c["column"]
        )
    )


print("\nMISSING FK ENDPOINTS")
print("=" * 70)

valid_pairs = []

for fk in fks:

    source = node_id(
        fk["db_id"],
        fk["from_table"],
        fk["from_column"]
    )

    target = node_id(
        fk["db_id"],
        fk["to_table"],
        fk["to_column"]
    )

    source_exists = source in exact_nodes
    target_exists = target in exact_nodes

    if not source_exists or not target_exists:

        print("\nFK:")
        print(source, "->", target)

        print("Exact source exists:", source_exists)
        print("Exact target exists:", target_exists)

        if not source_exists:
            key = (
                fk["db_id"].casefold(),
                fk["from_table"].casefold(),
                fk["from_column"].casefold()
            )

            print(
                "Case-insensitive source matches:",
                casefold_lookup.get(key, [])
            )

        if not target_exists:
            key = (
                fk["db_id"].casefold(),
                fk["to_table"].casefold(),
                fk["to_column"].casefold()
            )

            print(
                "Case-insensitive target matches:",
                casefold_lookup.get(key, [])
            )

            print("Columns actually present in target table:")

            for c in columns:
                if (
                    c["db_id"].casefold()
                    == fk["db_id"].casefold()
                    and
                    c["table"].casefold()
                    == fk["to_table"].casefold()
                ):
                    print("   ", c["column"])

    else:
        # Treat an undirected FK pair in canonical order
        pair = tuple(sorted([source, target]))
        valid_pairs.append(pair)


print("\n\nDUPLICATE UNDIRECTED FK PAIRS")
print("=" * 70)

counts = defaultdict(int)

for pair in valid_pairs:
    counts[pair] += 1

duplicates = 0

for pair, count in counts.items():
    if count > 1:
        duplicates += count - 1

        print("\nCount:", count)
        print(pair[0])
        print("<->")
        print(pair[1])

print("\nExtra duplicate FK records:", duplicates)
