import json
from statistics import mean

FILE = "features/spider/candidate_column_pairs.jsonl"

rows = []

with open(FILE) as f:
    for line in f:
        rows.append(json.loads(line))

gold = [r for r in rows if r["gold_join"] == 1]
nongold = [r for r in rows if r["gold_join"] == 0]

features = [
    "name_similarity",
    "jaccard",
    "containment_a_in_b",
    "containment_b_in_a",
    "min_overlap",
    "uniqueness_a",
    "uniqueness_b"
]

print("Gold joins:", len(gold))
print("Non-gold:", len(nongold))

print("\nAVERAGE FEATURE VALUES")
print("=" * 65)

for feature in features:
    g = mean(r[feature] for r in gold)
    n = mean(r[feature] for r in nongold)

    print(
        f"{feature:25} "
        f"gold={g:.3f}   non-gold={n:.3f}"
    )

print("\nBOOLEAN FEATURES")
print("=" * 65)

for feature in [
    "exact_name_match",
    "exact_type_match",
    "pk_a",
    "pk_b"
]:
    g = sum(bool(r[feature]) for r in gold) / len(gold)
    n = sum(bool(r[feature]) for r in nongold) / len(nongold)

    print(
        f"{feature:25} "
        f"gold={g:.3f}   non-gold={n:.3f}"
    )

print("\nSTRONG VALUE SIGNALS")
print("=" * 65)

for threshold in [0.5, 0.8, 1.0]:

    g = sum(
        r["min_overlap"] >= threshold
        for r in gold
    ) / len(gold)

    n = sum(
        r["min_overlap"] >= threshold
        for r in nongold
    ) / len(nongold)

    print(
        f"min_overlap >= {threshold}: "
        f"gold={g:.3f} non-gold={n:.3f}"
    )
