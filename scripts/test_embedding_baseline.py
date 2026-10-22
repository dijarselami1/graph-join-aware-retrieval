import json
import numpy as np

TRAIN = "features/spider/splits/train_pairs_embeddings.jsonl"
TEST = "features/spider/splits/test_pairs_embeddings.jsonl"


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(rows, threshold):
    tp = fp = fn = 0

    for r in rows:
        predicted = r["embedding_similarity"] >= threshold
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
        if precision + recall else 0
    )

    return precision, recall, f1, tp, fp, fn


train = load(TRAIN)
test = load(TEST)

# Find best threshold using TRAIN only
best = None

for threshold in np.arange(0.0, 1.001, 0.01):
    result = evaluate(train, threshold)

    if best is None or result[2] > best["f1"]:
        best = {
            "threshold": threshold,
            "precision": result[0],
            "recall": result[1],
            "f1": result[2],
            "tp": result[3],
            "fp": result[4],
            "fn": result[5]
        }

print("\nBEST EMBEDDING THRESHOLD ON TRAIN")
print("=" * 50)

for k, v in best.items():
    print(f"{k}: {v}")

# Test frozen threshold
result = evaluate(test, best["threshold"])

print("\nUNSEEN TEST DATABASES")
print("=" * 50)

print("Precision:", round(result[0], 3))
print("Recall:", round(result[1], 3))
print("F1:", round(result[2], 3))
print("True positives:", result[3])
print("False positives:", result[4])
print("Missed gold joins:", result[5])
