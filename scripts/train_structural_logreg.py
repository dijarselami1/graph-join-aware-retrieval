import json
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

TRAIN = "features/spider/splits/train_pairs.jsonl"
TEST = "features/spider/splits/test_pairs.jsonl"

FEATURES = [
    "name_similarity",
    "jaccard",
    "containment_a_in_b",
    "containment_b_in_a",
    "min_overlap",
    "uniqueness_a",
    "uniqueness_b",
    "exact_name_match",
    "exact_type_match",
    "pk_a",
    "pk_b",
]


def load(path):
    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    X = []
    y = []

    for r in rows:

        X.append([
            float(r["name_similarity"]),
            float(r["jaccard"]),
            float(r["containment_a_in_b"]),
            float(r["containment_b_in_a"]),
            float(r["min_overlap"]),
            float(r["uniqueness_a"]),
            float(r["uniqueness_b"]),
            int(r["exact_name_match"]),
            int(r["exact_type_match"]),
            int(r["pk_a"]),
            int(r["pk_b"]),
        ])

        y.append(int(r["gold_join"]))

    return np.array(X), np.array(y)


X_train, y_train = load(TRAIN)
X_test, y_test = load(TEST)


# --------------------------------------------------
# Train logistic regression
# --------------------------------------------------

model = Pipeline([
    ("scale", StandardScaler()),
    (
        "logreg",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)


# --------------------------------------------------
# Find best probability threshold using TRAIN only
# --------------------------------------------------

train_probs = model.predict_proba(X_train)[:, 1]

best_threshold = 0.5
best_f1 = -1

for threshold in np.arange(0.05, 0.96, 0.01):

    pred = (train_probs >= threshold).astype(int)

    f1 = f1_score(y_train, pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


def evaluate(name, X, y):

    probs = model.predict_proba(X)[:, 1]

    pred = (
        probs >= best_threshold
    ).astype(int)

    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    print(f"\n{name}")
    print("=" * 50)

    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))

    print("True positives:", tp)
    print("False positives:", fp)
    print("Missed gold joins:", fn)


print("\nSTRUCTURAL LOGISTIC REGRESSION")
print("=" * 50)

print(
    "Chosen probability threshold:",
    round(best_threshold, 2)
)

evaluate(
    "TRAIN DATABASES",
    X_train,
    y_train
)

evaluate(
    "UNSEEN TEST DATABASES",
    X_test,
    y_test
)


# --------------------------------------------------
# Show learned feature weights
# --------------------------------------------------

logreg = model.named_steps["logreg"]

coefficients = logreg.coef_[0]

results = sorted(
    zip(FEATURES, coefficients),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\nLEARNED FEATURE WEIGHTS")
print("=" * 50)

for feature, weight in results:

    direction = (
        "supports join"
        if weight > 0
        else "against join"
    )

    print(
        f"{feature:25} "
        f"{weight:8.3f}   "
        f"{direction}"
    )
