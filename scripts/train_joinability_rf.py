import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

ROOT = Path("/scratch/ds7491/graph_project")
PAIR_DIR = ROOT / "features/spider/all_pairs"

# --------------------------------------------------
# Load data
# --------------------------------------------------

X = np.load(PAIR_DIR / "features.npy", mmap_mode="r")
pairs = np.load(PAIR_DIR / "pair_indices.npy", mmap_mode="r")
labels = np.load(PAIR_DIR / "labels.npy", mmap_mode="r")

with open(PAIR_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

with open(ROOT / "cache/spider/column_cache.pkl", "rb") as f:
    cols = pickle.load(f)

with open(ROOT / "features/spider/splits/database_split.json") as f:
    split = json.load(f)

train_dbs = set(split["train_databases"])
test_dbs = set(split["test_databases"])

# --------------------------------------------------
# Keep only known labels: 0 or 1
# --------------------------------------------------

known_idx = np.where(labels >= 0)[0]

pair_a = pairs[known_idx, 0]

pair_db = np.array([
    cols[int(i)]["db_id"]
    for i in pair_a
])

train_mask = np.array([db in train_dbs for db in pair_db])
test_mask = np.array([db in test_dbs for db in pair_db])

train_idx = known_idx[train_mask]
test_idx = known_idx[test_mask]

print("Training pairs:", len(train_idx))
print("Training positives:", np.sum(labels[train_idx] == 1))

print("Test pairs:", len(test_idx))
print("Test positives:", np.sum(labels[test_idx] == 1))

# --------------------------------------------------
# Split training databases again:
# training + validation
# --------------------------------------------------

rng = np.random.RandomState(42)

train_db_list = sorted(train_dbs)
rng.shuffle(train_db_list)

val_count = max(1, int(len(train_db_list) * 0.20))

val_dbs = set(train_db_list[:val_count])
model_train_dbs = set(train_db_list[val_count:])

model_train_mask = np.array([
    db in model_train_dbs
    for db in pair_db[train_mask]
])

val_mask = np.array([
    db in val_dbs
    for db in pair_db[train_mask]
])

model_train_idx = train_idx[model_train_mask]
val_idx = train_idx[val_mask]

print("\nModel training pairs:", len(model_train_idx))
print("Validation pairs:", len(val_idx))

# --------------------------------------------------
# Train model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

print("\nTraining model...")

model.fit(
    X[model_train_idx],
    labels[model_train_idx]
)

# --------------------------------------------------
# Pick threshold using validation databases
# --------------------------------------------------

val_prob = model.predict_proba(X[val_idx])[:, 1]
val_y = labels[val_idx]

precision, recall, thresholds = precision_recall_curve(
    val_y,
    val_prob
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

best = np.argmax(f1)
threshold = thresholds[best]

print("\nBest validation threshold:", round(float(threshold), 4))
print("Validation F1:", round(float(f1[best]), 4))

# --------------------------------------------------
# Final untouched test
# --------------------------------------------------

test_prob = model.predict_proba(X[test_idx])[:, 1]
test_y = labels[test_idx]

test_pred = (test_prob >= threshold).astype(int)

print("\nTEST RESULTS")
print("Precision:", round(precision_score(test_y, test_pred), 4))
print("Recall:", round(recall_score(test_y, test_pred), 4))
print("F1:", round(f1_score(test_y, test_pred), 4))
print("Average precision:", round(
    average_precision_score(test_y, test_prob), 4
))

print("TP:", int(np.sum((test_pred == 1) & (test_y == 1))))
print("FP:", int(np.sum((test_pred == 1) & (test_y == 0))))
print("FN:", int(np.sum((test_pred == 0) & (test_y == 1))))

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

print("\nFEATURE IMPORTANCE")

importance = sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

for name, score in importance:
    print(f"{name:30s} {score:.4f}")

# --------------------------------------------------
# Save trained model + chosen threshold
# --------------------------------------------------

import joblib

MODEL_DIR = ROOT / "models/spider"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "joinability_rf.joblib"
)

with open(MODEL_DIR / "joinability_rf_meta.json", "w") as f:
    json.dump({
        "threshold": float(threshold),
        "features": feature_names
    }, f, indent=2)

print("\nSaved model to:", MODEL_DIR / "joinability_rf.joblib")
print("Saved threshold:", float(threshold))
