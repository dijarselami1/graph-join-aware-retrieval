import json

FILE = "features/spider/candidate_column_pairs.jsonl"

rows = []

with open(FILE) as f:
    for line in f:
        rows.append(json.loads(line))


def evaluate(name, rule):
    tp = fp = fn = 0

    for r in rows:
        predicted = rule(r)
        actual = r["gold_join"] == 1

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(
        f"{name:45} "
        f"P={precision:.3f} "
        f"R={recall:.3f} "
        f"F1={f1:.3f} "
        f"TP={tp} FP={fp}"
    )


print("\nFEATURE COMBINATION TESTS")
print("=" * 90)

# Single signals
evaluate(
    "Exact name match",
    lambda r: r["exact_name_match"]
)

evaluate(
    "Min overlap >= 0.8",
    lambda r: r["min_overlap"] >= 0.8
)

evaluate(
    "Name similarity >= 0.8",
    lambda r: r["name_similarity"] >= 0.8
)

# Combinations
evaluate(
    "Exact name + overlap >= 0.8",
    lambda r:
        r["exact_name_match"]
        and r["min_overlap"] >= 0.8
)

evaluate(
    "Name sim >= .8 + overlap >= .8",
    lambda r:
        r["name_similarity"] >= 0.8
        and r["min_overlap"] >= 0.8
)

evaluate(
    "Name + overlap + type match",
    lambda r:
        r["name_similarity"] >= 0.8
        and r["min_overlap"] >= 0.8
        and r["exact_type_match"]
)

evaluate(
    "Name + overlap + PK evidence",
    lambda r:
        r["name_similarity"] >= 0.8
        and r["min_overlap"] >= 0.8
        and (r["pk_a"] or r["pk_b"])
)

evaluate(
    "Name + overlap + type + PK",
    lambda r:
        r["name_similarity"] >= 0.8
        and r["min_overlap"] >= 0.8
        and r["exact_type_match"]
        and (r["pk_a"] or r["pk_b"])
)
