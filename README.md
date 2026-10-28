# Graph-Based Join-Aware Table Selection for Multi-Table Question Answering

NYU Abu Dhabi capstone extending the DBRAG research direction. Relevance-based
retrieval can find endpoint tables while missing intermediate tables needed to
connect them. This repository investigates recovering those bridge tables from
predicted column joins, then selecting paths relevant to a question. It currently
accepts **two manually supplied seeds**; it is not an integrated DBRAG QA system.

## Architecture and existing graph construction

1. `extract_spider_metadata.py` extracts SQLite schemas, keys, statistics and samples.
   `build_spider_gold_graph.py` and `build_spider_gold_hetero_graph.py` create gold
   reference graphs; the latter connects tables to columns with membership edges.
2. `build_candidate_features.py` and `split_train_test_databases.py` build the
   earlier within-database experiment and a deterministic database split.
   `add_embedding_similarity.py` generates local all-MiniLM-L6-v2 embeddings.
3. `build_column_cache.py` caches normalized distinct column values.
   `build_all_pair_features.py` computes all column pairs across databases using
   11 features: embedding cosine, column and table name similarity, value overlap,
   Jaccard, two primary-key flags, two uniqueness ratios, datatype compatibility,
   and distinct-count ratio.
4. `build_pair_labels.py` labels known Spider foreign keys positive, other
   same-database pairs negative, and cross-database pairs unknown. These negatives
   are a supervision assumption, not proof of non-joinability.
5. `train_joinability_rf.py` trains a 500-tree Random Forest using database splits,
   selects a threshold on validation databases, and evaluates held-out databases.
   `score_all_pairs.py` applies it globally. The committed threshold metadata is
   `0.5160912733321336`.
6. `export_predicted_edges.py` retains edges above that threshold and projects
   column predictions into an undirected table graph, retaining the strongest
   column join per table pair. Cross-database edges are retained.
7. `query_driven_paths.py` generates simple paths using a priority queue ordered by
   sum of negative log edge scores (equivalently decreasing product). It then
   annotates the fixed candidate set with separate lexical, embedding and optional
   GPT-4o-mini rankings. Candidate generation remains graph-only.

An earlier global top-50 embedding/structural-rule pipeline and structural,
embedding and logistic-regression baselines remain in `scripts/`. They are
separate from the all-pairs Random Forest pipeline. `find_graph_path.py` remains
unchanged as the original single-best-path demo.

## Local use

The path search, lexical reranker and new tests use only the Python standard
library, with Python 3.6-compatible source. No API key, package installation or
model download is required for these commands:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/query_driven_paths.py \
  --edges tests/fixtures/query_edges.csv \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --max-hops 5 --top-k 8 --ranking query-aware \
  --query-weight 0.5 --length-penalty 0 \
  --output results/synthetic_query.json
```

The fixture is synthetic, not a subset of measured predictions. Add “concert” to
the question to see a lexical reranking example. Use `--ranking graph` (the
backward-compatible default) for the graph-only baseline. Both modes export both
ranks and scores. `--same-database` is an optional Spider ablation, off by default.
`--max-states 150000` bounds popped search states, not heap memory. `--top-k` is
candidate count **before** reranking, and `--max-hops` counts edges.

Question relevance is the fraction of distinct normalized question tokens found
in path table names or joining column names. Database names are excluded. The
combined score is `((1-alpha)*G + alpha*R)*exp(-lambda*max(0,hops-1))`, where `G`
is the edge-score product. Alpha defaults to 0.5 as an explicit untuned equal
mixture; lambda defaults to zero because products already favor shorter paths.
No candidate-dependent normalization is used. This is lexical matching, not
semantic understanding or LLM-based relevance.

The optional `--llm` adapter now produces a validated full ranking plus a selection
or `NONE`. It is separate from embeddings and lexical matching and is never
enabled by `--compare` alone. It requires an API key and can incur charges.
No API calls were made for implementation or testing.

## Jubail

With the updated files copied into the existing project directory, and the
predicted table-edge CSV already available:

```bash
cd /scratch/ds7491/graph_project
source .venv/bin/activate
python -m unittest discover -s tests -v
mkdir -p logs results
sbatch query_demo.sbatch
```

The batch job writes `results/query_graph_baseline.json` and
`results/query_aware_paths.json` using identical candidates and parameters.
`logs/` must exist **before** submission for Slurm output redirection. This uses
the existing environment, not a newly resolved dependency stack. The Python 3.6
runtime itself was not available locally for execution testing.

For new graph construction, follow the staged commands and prerequisites in
[EXPERIMENTS.md](docs/EXPERIMENTS.md). Most existing construction scripts hardcode
`/scratch/ds7491/graph_project`; the query script accepts any edge CSV path.
Training/feature extraction need the existing NumPy, scikit-learn, RapidFuzz,
SentenceTransformers, NetworkX and joblib environment. This checkout does not
include Spider databases, caches, embeddings, feature arrays, trained model
weights, or predicted CSVs; it includes gold GraphML, model metadata and historical
logs. Do not expect a full training run to work from the checkout alone.

## Outputs, limitations and next steps

JSON includes question, seeds, run parameters, truncation and missing-seed status,
path tables, intermediate tables, oriented column joins, individual edge scores,
raw graph score, lexical evidence, query-aware score and both ranks. See
[QUERY_AWARE_RETRIEVAL.md](docs/QUERY_AWARE_RETRIEVAL.md) for equations and schema.

The reranker cannot recover paths omitted by the graph candidate budget. Shared
seed tokens often make relevance identical across paths; “performed” is not
mapped to “concert”. Names may be ambiguous, the plural rule is simplistic, and
predicted joins may be invalid. Edge products are ranking heuristics, not calibrated
path probabilities. The strongest-column projection loses alternative joins and
composite-key structure. End-to-end QA accuracy and cross-source join validity
have not been evaluated here.

Planned work: annotate question/path relevance, evaluate intermediate-table recall
and ranking, validate joins by execution, tune weights on held-out validation
questions, evaluate the implemented semantic/LLM rerankers on real questions,
and integrate with DBRAG. Interactive
visualization integration is future work; no visualization repository was changed.
[EXPERIMENTS.md](docs/EXPERIMENTS.md) separates historical logs from local checks;
[DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) records this change.

## Separate semantic and LLM methods (implemented)

`--ranking lexical` aliases the existing `query-aware` mode. `--ranking embedding`
uses a local SentenceTransformer; `--compare` computes graph, lexical and embedding
rankings on one candidate set. Neither flag enables GPT. `--llm` independently
requests GPT-4o-mini assessments; `--ranking llm --llm` selects that output order.
The JSON `rankings` object provides comparable ordered IDs and method statuses.
Unrequested methods have null rankings, not fabricated results.

The recommended initial model is the same family already used for column
embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Supply a complete, previously
saved local SentenceTransformer directory. Loading uses `local_files_only=True`
and disables remote model code; this command does not download weights. See the
[SentenceTransformer interface](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

Graph/lexical/LLM transport and offline tests retain Python 3.6-compatible source.
The **embedding runtime** additionally needs compatible PyTorch and
SentenceTransformers versions supporting `local_files_only`. Use a separate modern
Python environment on Jubail if the existing environment is too old. No ML
packages or weights were installed during this change; an actual embedding runtime
has not been validated here. Preserve the environment package versions and model
snapshot/checksum for experiments.

```bash
# Set this to your existing, complete local MiniLM model directory.
export EMBEDDING_MODEL=/absolute/path/to/all-MiniLM-L6-v2
python scripts/query_driven_paths.py \
  --edges tests/fixtures/query_edges.csv \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --compare --ranking embedding --embedding-model "$EMBEDDING_MODEL" \
  --embedding-weight 0.5 --length-penalty 0 \
  --output results/semantic_comparison.json
```

Embedding scoring uses `E=max(0, cosine(question_embedding, path_embedding))` and
`S_embedding=((1-beta)*G+beta*E)*L`. Beta defaults to an untuned 0.5. This preserves
raw cosine, clips negative similarity to no positive evidence, and never mixes in
the lexical score. Path texts contain table names and explicit proposed joins,
without database prefixes or graph scores. Long texts may be truncated by the
model; semantic similarity does not prove join validity.

For Jubail, set the local model directory and optionally the embedding environment:

```bash
cd /scratch/ds7491/graph_project
mkdir -p logs results
export EMBEDDING_MODEL=/absolute/path/to/all-MiniLM-L6-v2
# Only if a separate environment is needed:
export EMBEDDING_VENV=/absolute/path/to/embedding-venv
sbatch query_compare.sbatch
```

The job produces `results/query_comparison.json` without API calls. For a later,
explicitly approved live LLM experiment, add `--llm` to a CLI comparison command
with `OPENAI_API_KEY` set in your environment. To isolate GPT without loading
embeddings, use `--ranking llm --llm` without `--compare`. Do not place keys in
source files or output JSON. GPT receives the question and proposed table/column
joins, without scores from the other methods. It returns brief assessments and
uncertainty, not a proof that the joins execute. The adapter uses GPT-4o-mini's
[structured output format](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)
and additionally validates candidate membership, completeness and selection.

**Verified:** 31 offline tests, including controlled vectors and mocked HTTP.
**Not verified:** actual MiniLM rankings, live GPT responses, full Spider retrieval
quality or QA gains. See the experiment log for the distinction between synthetic
mechanics and future model evaluation.
