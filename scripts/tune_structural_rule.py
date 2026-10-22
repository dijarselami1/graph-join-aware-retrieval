import json

TRAIN = "features/spider/splits/train_pairs.jsonl"
TEST = "features/spider/splits/test_pairs.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


train = load(TRAIN)
test = load(TEST)


def evaluate(rows, name_t, overlap_t, require_pk, require_type):
    tp = fp = fn = 0

    for r in rows:

        predicted = (
            r["name_similarity"] >= name_t
            and r["min_overlap"] >= overlap_t
        )

        if require_pk:
            predicted = predicted and (r["pk_a"] or r["pk_b"])

        if require_type:
            predicted = predicted and r["exact_type_match"]

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

    return precision, recall, f1, tp, fp, fn


# ------------------------------------------
# Find best rule using TRAIN ONLY
# ------------------------------------------

best = None

thresholds = [
    0.4, 0.5, 0.6, 0.7,
    0.75, 0.8, 0.85, 0.9,
    0.95, 1.0
]

for name_t in thresholds:
    for overlap_t in thresholds:
        for require_pk in [False, True]:
            for require_type in [False, True]:

                result = evaluate(
                    train,
                    name_t,
                    overlap_t,
                    require_pk,
                    require_type
                )

                p, r, f1, tp, fp, fn = result

                if best is None or f1 > best["f1"]:
                    best = {
                        "name_threshold": name_t,
                        "overlap_threshold": overlap_t,
                        "require_pk": require_pk,
                        "require_type": require_type,
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn
                    }


print("\nBEST RULE FOUND ON TRAIN")
print("=" * 50)

for k, v in best.items():
    print(f"{k}: {v}")


# ------------------------------------------
# Freeze rule and evaluate on TEST
# ------------------------------------------

test_result = evaluate(
    test,
    best["name_threshold"],
    best["overlap_threshold"],
    best["require_pk"],
    best["require_type"]
)

p, r, f1, tp, fp, fn = test_result

print("\nUNSEEN TEST DATABASES")
print("=" * 50)

print("Precision:", round(p, 3))
print("Recall:", round(r, 3))
print("F1:", round(f1, 3))

print("True positives:", tp)
print("False positives:", fp)
print("Missed gold joins:", fn)
