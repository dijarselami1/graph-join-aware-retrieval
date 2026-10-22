import json
from pathlib import Path
import networkx as nx

ROOT = Path("/scratch/ds7491/graph_project")
META_DIR = ROOT / "metadata" / "spider"
GRAPH_DIR = ROOT / "graphs" / "spider"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load our extracted Spider metadata
# -------------------------------------------------

with open(META_DIR / "columns.json") as f:
    columns = json.load(f)

with open(META_DIR / "foreign_keys.json") as f:
    foreign_keys = json.load(f)


def node_id(db_id, table, column):
    """
    Gives every column a unique ID.
    Example:
    concert_singer##concert##Stadium_ID
    """
    return f"{db_id}##{table}##{column}"


# -------------------------------------------------
# Create the gold graph
# -------------------------------------------------

# Undirected because for join/path reasoning we want to
# be able to travel in either direction across a FK.
G_gold = nx.Graph()


# -------------------------------------------------
# Add COLUMN nodes
# -------------------------------------------------

for col in columns:

    nid = node_id(
        col["db_id"],
        col["table"],
        col["column"]
    )

    G_gold.add_node(
        nid,

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


# -------------------------------------------------
# Add GOLD foreign-key edges
# -------------------------------------------------

missing_nodes = 0

for fk in foreign_keys:

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

    if source not in G_gold or target not in G_gold:
        print("WARNING: missing node for FK:")
        print(source, "->", target)
        missing_nodes += 1
        continue

    G_gold.add_edge(
        source,
        target,

        edge_type="foreign_key",

        # Keep the original FK direction as metadata
        fk_table=fk["from_table"],
        fk_column=fk["from_column"],

        referenced_table=fk["to_table"],
        referenced_column=fk["to_column"],

        db_id=fk["db_id"]
    )


# -------------------------------------------------
# Save graph
# -------------------------------------------------

output_file = GRAPH_DIR / "gold_column_graph.graphml"

nx.write_graphml(
    G_gold,
    output_file
)


# -------------------------------------------------
# Print summary
# -------------------------------------------------

print("\nGOLD SPIDER COLUMN GRAPH")
print("=" * 40)

print("Nodes:", G_gold.number_of_nodes())
print("Edges:", G_gold.number_of_edges())
print("Original FK records:", len(foreign_keys))
print("Missing FK nodes:", missing_nodes)

print(
    "Connected components:",
    nx.number_connected_components(G_gold)
)

isolated_nodes = list(nx.isolates(G_gold))

print("Isolated columns:", len(isolated_nodes))

print("\nSaved graph:")
print(output_file)


print("\nFirst 5 gold edges:")

for source, target, data in list(
    G_gold.edges(data=True)
)[:5]:

    print()
    print(source)
    print("   <---- GOLD FK ---->")
    print(target)


# -------------------------------------------------
# Sanity check on concert_singer
# -------------------------------------------------

print("\nconcert_singer GOLD edges:")
print("-" * 40)

for source, target in G_gold.edges():

    if (
        G_gold.nodes[source]["db_id"]
        == "concert_singer"
    ):

        print(
            G_gold.nodes[source]["table"]
            + "."
            + G_gold.nodes[source]["column"],
            "<->",
            G_gold.nodes[target]["table"]
            + "."
            + G_gold.nodes[target]["column"]
        )
