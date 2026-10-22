import json
import numpy as np

TRAIN = "features/spider/splits/train_pairs_embeddings.jsonl"
TEST = "features/spider/splits/test_pairs_embeddings.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def structural_rule(r):
    return (
        r["name_similarity"] >= 0.85
        and r["min_overlap"] >= 0.85
        and (r["pk_a"] or r["pk_b"])
    )


def evaluate(rows, embedding_threshold=None):

    tp = fp = fn = 0

    for r in rows:

        predicted = structural_rule(r)

        # If using embedding as an extra filter
        if embedding_threshold is not None:
            predicted = (
                predicted
                and r["embedding_similarity"] >= embedding_threshold
            )

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


# --------------------------------------------------
# Baseline structural rule
# --------------------------------------------------

base = evaluate(test)

print("\nSTRUCTURAL ONLY - TEST")
print("=" * 50)
print("Precision:", round(base[0], 3))
print("Recall:", round(base[1], 3))
print("F1:", round(base[2], 3))
print("TP:", base[3])
print("FP:", base[4])
print("FN:", base[5])


# --------------------------------------------------
# Tune embedding threshold using TRAIN only
# --------------------------------------------------

best = None

for threshold in np.arange(0.0, 1.001, 0.01):

    result = evaluate(train, threshold)

    if best is None or result[2] > best["f1"]:
        best = {
            "threshold": round(float(threshold), 2),
            "precision": result[0],
            "recall": result[1],
            "f1": result[2],
            "tp": result[3],
            "fp": result[4],
            "fn": result[5]
        }


print("\nBEST STRUCTURAL + EMBEDDING ON TRAIN")
print("=" * 50)

for k, v in best.items():
    print(f"{k}: {v}")


# --------------------------------------------------
# Frozen threshold on TEST
# --------------------------------------------------

result = evaluate(test, best["threshold"])

print("\nSTRUCTURAL + EMBEDDING - UNSEEN TEST")
print("=" * 50)

print("Embedding threshold:", best["threshold"])
print("Precision:", round(result[0], 3))
print("Recall:", round(result[1], 3))
print("F1:", round(result[2], 3))
print("TP:", result[3])
print("FP:", result[4])
print("FN:", result[5])
