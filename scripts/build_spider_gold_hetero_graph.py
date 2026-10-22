import json
from pathlib import Path
from collections import defaultdict
import networkx as nx

ROOT = Path("/scratch/ds7491/graph_project")
META = ROOT / "metadata" / "spider"
GRAPH_DIR = ROOT / "graphs" / "spider"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

with open(META / "columns.json") as f:
    columns = json.load(f)

with open(META / "foreign_keys.json") as f:
    foreign_keys = json.load(f)


def table_id(db, table):
    return f"table::{db}##{table}"


def column_id(db, table, column):
    return f"column::{db}##{table}##{column}"


# -------------------------------------------------
# Build lookup for case-insensitive matching
# -------------------------------------------------

column_lookup = {}

for col in columns:
    key = (
        col["db_id"].casefold(),
        col["table"].casefold(),
        col["column"].casefold()
    )

    column_lookup[key] = column_id(
        col["db_id"],
        col["table"],
        col["column"]
    )


# -------------------------------------------------
# Graph
# -------------------------------------------------

G = nx.Graph()

table_nodes = set()


# -------------------------------------------------
# Add table nodes + column nodes + membership edges
# -------------------------------------------------

for col in columns:

    tid = table_id(
        col["db_id"],
        col["table"]
    )

    cid = column_id(
        col["db_id"],
        col["table"],
        col["column"]
    )

    if tid not in table_nodes:

        G.add_node(
            tid,
            node_type="table",
            db_id=col["db_id"],
            table=col["table"]
        )

        table_nodes.add(tid)

    G.add_node(
        cid,
        node_type="column",

        db_id=col["db_id"],
        table=col["table"],
        column=col["column"],

        datatype=str(col["datatype"]),

        row_count=int(col["row_count"]),
        distinct_count=int(col["distinct_count"]),

        uniqueness_ratio=float(
            col["uniqueness_ratio_rows"]
        ),

        is_primary_key=bool(
            col["is_primary_key"]
        ),

        primary_key_position=int(
            col["primary_key_position"]
        )
    )

    # Explicitly record:
    # THIS COLUMN BELONGS TO THIS TABLE
    G.add_edge(
        tid,
        cid,
        edge_type="contains"
    )


# -------------------------------------------------
# Add gold FK/join edges
# -------------------------------------------------

unresolved = []
resolved_fk_records = 0
duplicate_gold_edges = 0

seen_gold_pairs = set()


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

    pair = tuple(sorted([source, target]))

    if pair in seen_gold_pairs:
        duplicate_gold_edges += 1
        continue

    seen_gold_pairs.add(pair)

    G.add_edge(
        source,
        target,

        edge_type="gold_join",

        db_id=fk["db_id"],

        fk_table=fk["from_table"],
        fk_column=fk["from_column"],

        referenced_table=fk["to_table"],
        referenced_column=fk["to_column"]
    )

    resolved_fk_records += 1


# -------------------------------------------------
# Save graph
# -------------------------------------------------

graph_file = GRAPH_DIR / "gold_heterogeneous_graph.graphml"

nx.write_graphml(
    G,
    graph_file
)


# Save unresolved FKs separately
with open(
    GRAPH_DIR / "unresolved_foreign_keys.json",
    "w"
) as f:
    json.dump(unresolved, f, indent=2)


# -------------------------------------------------
# Summary
# -------------------------------------------------

table_count = sum(
    1 for _, d in G.nodes(data=True)
    if d["node_type"] == "table"
)

column_count = sum(
    1 for _, d in G.nodes(data=True)
    if d["node_type"] == "column"
)

membership_edges = sum(
    1 for _, _, d in G.edges(data=True)
    if d["edge_type"] == "contains"
)

gold_join_edges = sum(
    1 for _, _, d in G.edges(data=True)
    if d["edge_type"] == "gold_join"
)


print("\nGOLD HETEROGENEOUS SPIDER GRAPH")
print("=" * 50)

print("Table nodes:", table_count)
print("Column nodes:", column_count)

print("\nMembership edges:", membership_edges)
print("Gold join edges:", gold_join_edges)

print("\nOriginal FK records:", len(foreign_keys))
print("Duplicate FK records skipped:", duplicate_gold_edges)
print("Unresolved FK records:", len(unresolved))

print("\nSaved graph:")
print(graph_file)

print("\nUnresolved FKs saved to:")
print(
    GRAPH_DIR /
    "unresolved_foreign_keys.json"
)


# -------------------------------------------------
# concert_singer example
# -------------------------------------------------

print("\nconcert_singer gold joins:")
print("-" * 50)

for u, v, data in G.edges(data=True):

    if (
        data["edge_type"] == "gold_join"
        and data["db_id"] == "concert_singer"
    ):

        print(
            G.nodes[u]["table"],
            ".",
            G.nodes[u]["column"],
            " <-> ",
            G.nodes[v]["table"],
            ".",
            G.nodes[v]["column"],
            sep=""
        )
