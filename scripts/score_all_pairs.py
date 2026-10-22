import json
import joblib
import numpy as np
from pathlib import Path

ROOT = Path("/scratch/ds7491/graph_project")

X = np.load(
    ROOT / "features/spider/all_pairs/features.npy",
    mmap_mode="r"
)

model = joblib.load(
    ROOT / "models/spider/joinability_rf.joblib"
)

with open(ROOT / "models/spider/joinability_rf_meta.json") as f:
    meta = json.load(f)

threshold = meta["threshold"]

scores = np.lib.format.open_memmap(
    ROOT / "features/spider/all_pairs/joinability_scores.npy",
    mode="w+",
    dtype=np.float32,
    shape=(len(X),)
)

chunk = 200000

print("Pairs to score:", len(X))
print("Threshold:", threshold)

for start in range(0, len(X), chunk):
    end = min(start + chunk, len(X))

    scores[start:end] = model.predict_proba(
        X[start:end]
    )[:, 1]

    print(f"Scored {end}/{len(X)}")

scores.flush()

predicted = int(np.sum(scores >= threshold))

print("\nDONE")
print("Total pairs scored:", len(scores))
print("Predicted joinable:", predicted)
print("Saved joinability_scores.npy")
