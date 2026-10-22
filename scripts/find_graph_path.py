import csv
import math
import heapq
import sys

EDGE_FILE = "/scratch/ds7491/graph_project/graphs/spider/predicted_table_edges.csv"

graph = {}

with open(EDGE_FILE) as f:
    reader = csv.DictReader(f)

    for row in reader:
        a = row["db_a"] + "." + row["table_a"]
        b = row["db_b"] + "." + row["table_b"]
        score = float(row["score"])

        info = {
            "score": score,
            "column_a": row["column_a"],
            "column_b": row["column_b"]
        }

        graph.setdefault(a, []).append((b, info))
        graph.setdefault(b, []).append((a, {
            "score": score,
            "column_a": row["column_b"],
            "column_b": row["column_a"]
        }))


def best_path(start, end):
    # -log(score) means shortest path = highest product of edge scores
    pq = [(0.0, start, [])]
    best = {}

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in best and cost >= best[node]:
            continue

        best[node] = cost
        new_path = path + [node]

        if node == end:
            return new_path

        for neighbor, info in graph.get(node, []):
            new_cost = cost - math.log(info["score"])
            heapq.heappush(pq, (new_cost, neighbor, new_path))

    return None


def get_edge(a, b):
    for neighbor, info in graph[a]:
        if neighbor == b:
            return info
    return None


start = sys.argv[1]
end = sys.argv[2]

path = best_path(start, end)

print("\nSeed tables:")
print(" ", start)
print(" ", end)

if path is None:
    print("\nNo path found.")
    sys.exit()

print("\nBest graph path:")

product = 1.0

for i in range(len(path) - 1):
    a = path[i]
    b = path[i + 1]
    edge = get_edge(a, b)

    product *= edge["score"]

    print(
        "{} [{}] -- {:.4f} --> [{}] {}".format(
            a,
            edge["column_a"],
            edge["score"],
            edge["column_b"],
            b
        )
    )

print("\nIntermediate tables:")
for node in path[1:-1]:
    print(" ", node)

print("\nPath confidence:", round(product, 4))
