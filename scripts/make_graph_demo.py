import csv

EDGE_FILE = "/scratch/ds7491/graph_project/graphs/spider/predicted_table_edges.csv"
OUT_FILE = "/scratch/ds7491/graph_project/graph_demo.dot"

PATH = [
    "concert_singer.singer",
    "concert_singer.singer_in_concert",
    "concert_singer.concert",
    "concert_singer.stadium"
]

SEEDS = {
    "concert_singer.singer",
    "concert_singer.stadium"
}

MAX_EXTRA_NEIGHBORS = 2

edges = []

with open(EDGE_FILE) as f:
    reader = csv.DictReader(f)

    for row in reader:
        a = row["db_a"] + "." + row["table_a"]
        b = row["db_b"] + "." + row["table_b"]

        edges.append({
            "a": a,
            "b": b,
            "score": float(row["score"]),
            "column_a": row["column_a"],
            "column_b": row["column_b"]
        })


# Find the actual path edges
path_pairs = set()

for i in range(len(PATH) - 1):
    path_pairs.add(frozenset([PATH[i], PATH[i + 1]]))


selected_edges = []
selected_nodes = set(PATH)


# Add the path edges first
for edge in edges:
    pair = frozenset([edge["a"], edge["b"]])

    if pair in path_pairs:
        selected_edges.append((edge, True))


# Add a few strongest alternative neighbors around each path node
for node in PATH:

    candidates = []

    for edge in edges:

        pair = frozenset([edge["a"], edge["b"]])

        if pair in path_pairs:
            continue

        if edge["a"] == node:
            neighbor = edge["b"]
        elif edge["b"] == node:
            neighbor = edge["a"]
        else:
            continue

        candidates.append((edge["score"], edge, neighbor))

    candidates.sort(reverse=True, key=lambda x: x[0])

    added = 0

    for score, edge, neighbor in candidates:

        if added >= MAX_EXTRA_NEIGHBORS:
            break

        selected_edges.append((edge, False))
        selected_nodes.add(neighbor)
        added += 1


def short_name(node):
    return node.split(".", 1)[1]


with open(OUT_FILE, "w") as f:

    f.write("graph G {\n")
    f.write("rankdir=LR;\n")
    f.write('graph [pad="0.4", nodesep="0.55", ranksep="0.9"];\n')
    f.write('node [shape=box, style="rounded,filled", fontname="Helvetica"];\n')
    f.write('edge [fontname="Helvetica"];\n\n')

    # Nodes
    for node in selected_nodes:

        label = short_name(node)

        if node in SEEDS:
            f.write(
                '"{}" [label="SEED\\n{}", fillcolor="#DDF3E4", '
                'color="#2E7D32", penwidth=2.5];\n'.format(node, label)
            )

        elif node in PATH:
            f.write(
                '"{}" [label="INTERMEDIATE\\n{}", fillcolor="#FFF0CC", '
                'color="#D48B00", penwidth=2.5];\n'.format(node, label)
            )

        else:
            f.write(
                '"{}" [label="{}", fillcolor="#F2F2F2", '
                'color="#AAAAAA"];\n'.format(node, label)
            )

    f.write("\n")

    # Edges
    seen = set()

    for edge, is_path in selected_edges:

        key = frozenset([edge["a"], edge["b"]])

        if key in seen:
            continue

        seen.add(key)

        label = "{} ↔ {}\\n{:.4f}".format(
            edge["column_a"],
            edge["column_b"],
            edge["score"]
        )

        if is_path:
            f.write(
                '"{}" -- "{}" [label="{}", color="#1565C0", '
                'fontcolor="#1565C0", penwidth=4];\n'.format(
                    edge["a"], edge["b"], label
                )
            )
        else:
            f.write(
                '"{}" -- "{}" [label="{:.3f}", color="#BBBBBB", '
                'fontcolor="#777777", penwidth=1];\n'.format(
                    edge["a"], edge["b"], edge["score"]
                )
            )

    f.write("\n")
    f.write('labelloc="t";\n')
    f.write('label="Graph-Based Path Inference Demo\\n')
    f.write('Seeds: singer + stadium | Recovered: singer_in_concert + concert | Path confidence: 0.8635";\n')
    f.write("}\n")


print("Saved:", OUT_FILE)
