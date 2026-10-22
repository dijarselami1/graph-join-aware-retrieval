import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path("/scratch/ds7491/graph_project")

COLUMNS_FILE = ROOT / "metadata/spider/columns.json"

TRAIN_FILE = ROOT / "features/spider/splits/train_pairs.jsonl"
TEST_FILE = ROOT / "features/spider/splits/test_pairs.jsonl"

OUT_TRAIN = ROOT / "features/spider/splits/train_pairs_embeddings.jsonl"
OUT_TEST = ROOT / "features/spider/splits/test_pairs_embeddings.jsonl"

EMBED_DIR = ROOT / "embeddings/spider"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def key(db, table, column):
    return (
        db.casefold(),
        table.casefold(),
        column.casefold()
    )


# --------------------------------------------------
# Load columns
# --------------------------------------------------

with open(COLUMNS_FILE) as f:
    columns = json.load(f)

texts = []
keys = []

for c in columns:

    samples = ", ".join(
        str(x) for x in c.get("sample_values", [])
    )

    # IMPORTANT:
    # db_id deliberately NOT included.
    text = (
        f"table: {c['table']} | "
        f"column: {c['column']} | "
        f"type: {c['datatype']} | "
        f"sample values: {samples}"
    )

    texts.append(text)

    keys.append(
        key(
            c["db_id"],
            c["table"],
            c["column"]
        )
    )


print("Columns to embed:", len(texts))
print("Model:", MODEL_NAME)

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

lookup = {
    k: i
    for i, k in enumerate(keys)
}

np.save(
    EMBED_DIR / "column_embeddings.npy",
    embeddings
)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# Add cosine similarity to candidate pairs
# --------------------------------------------------

def process(input_file, output_file):

    count = 0
    missing = 0

    with open(input_file) as fin, open(output_file, "w") as fout:

        for line in fin:

            r = json.loads(line)

            a = key(
                r["db_id"],
                r["table_a"],
                r["column_a"]
            )

            b = key(
                r["db_id"],
                r["table_b"],
                r["column_b"]
            )

            if a not in lookup or b not in lookup:
                missing += 1
                continue

            emb_a = embeddings[lookup[a]]
            emb_b = embeddings[lookup[b]]

            # Embeddings are normalized,
            # so dot product = cosine similarity
            cosine = float(
                np.dot(emb_a, emb_b)
            )

            r["embedding_similarity"] = round(
                cosine,
                6
            )

            fout.write(
                json.dumps(r) + "\n"
            )

            count += 1

    print()
    print(output_file.name)
    print("Pairs written:", count)
    print("Missing:", missing)


process(
    TRAIN_FILE,
    OUT_TRAIN
)

process(
    TEST_FILE,
    OUT_TEST
)

print("\nDONE")
