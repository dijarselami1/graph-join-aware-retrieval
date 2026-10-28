# Experiments and reproduction

## Evidence available in this checkout

The repository contains historical stdout logs, RF threshold/feature metadata,
gold GraphML files, unresolved FK records, and static graph demos. It does not
contain the predicted table CSV, model weights, Spider databases, feature arrays,
embeddings or query-path result JSONs. Logs below were inspected, not regenerated.
Their presence does not establish reproducibility of the original environment.

| Committed source | Recorded result |
| --- | --- |
| `column_cache.out`, `all_pairs.out` | 4,497 columns; 10,109,256 pairs; 11 features |
| `pair_labels.out` | 795 positive pairs; 135,937 same-database negatives; 9,972,524 cross-database unknowns; 3 unresolved FK records |
| `join_rf.out` | 42,189 model-training pairs; 15,089 validation pairs; 79,454 test pairs including 210 positives |
| `join_rf.out` | Validation F1 0.7192; test precision 0.8364, recall 0.6571, F1 0.7360, AP 0.8251; TP 138, FP 27, FN 72 |
| `models/spider/joinability_rf_meta.json` | Threshold 0.5160912733321336 and feature names |
| `score_all.out` | 10,109,256 scored pairs; 2,351 predicted joinable column pairs |

These are **column-pair classification results**, not table-path relevance or QA
accuracy. The label policy treats unlisted within-database relationships as
negative and cross-database relationships as unknown; do not count unknown
cross-source joins as false positives merely because Spider has no FK label.
The threshold was selected using validation databases in the RF script.

`demo_path.dot` shows singer → singer_in_concert → concert → stadium with rounded
edge labels 0.9206, 0.9866 and 0.9507 and an annotated path confidence of 0.8635.
`make_graph_demo.py` hardcodes that route and confidence label. This supports
reporting an existing illustrative artifact, not an independently verified search
result or an exact product of original scores. The user's initial successful
recovery report was not rerun against predicted Spider edges locally.

## Lightweight verification performed for this change

Run from repository root:

```bash
python3 -m unittest discover -s tests -v
```

The 15 standard-library tests pass on the local Python runtime. They cover bridge
recovery, products and mixture/penalty arithmetic, lexical normalization and column
evidence, successful reranking, equal-relevance limitations, graph-only mode,
cycles, hop/K/state limits, cross-database traversal and its optional constraint,
missing/disconnected seeds, identical seeds, reversed join columns, invalid
parameters, nonfinite scores, equal-score parallel joins, shuffled CSV order,
an exhaustive simple-path oracle, CLI JSON schema and repeated-run determinism.
No datasets, trained artifacts or APIs are required. The older `scripts/test_*.py`
files are dataset-dependent experiments, so use `-s tests` rather than importing
those scripts as a general test suite.

`tests/fixtures/query_edges.csv` contains five deliberately constructed edges:
three 0.9 bridge edges and two 0.95 archive edges crossing databases. With threshold
0.5160912733321336, H=5, K=8, state budget=150000, alpha=0.5 and lambda=0:

- The expected bridge is present at graph rank 2 with product 0.729.
- The unrelated route is graph rank 1 with product 0.9025.
- The original singer–stadium question gives both relevance 2/3; order is unchanged.
- Adding “concert” to the question gives bridge relevance 3/4 and archive relevance
  2/4. Scores become 0.7395 and 0.70125, respectively; the bridge moves to rank 1.
- Setting lambda=0.1 in that last case promotes the shorter archive again.

These tests demonstrate behavior on constructed inputs, not statistical gains.
No full Spider experiment, end-to-end QA evaluation, or paid API call was run.

## Paired path-ranking experiments on Jubail

Copy the changed script, batch file, tests and docs into the existing project,
then use the existing environment and predicted CSV:

```bash
cd /scratch/ds7491/graph_project
source .venv/bin/activate
python -m unittest discover -s tests -v
mkdir -p logs results
sbatch query_demo.sbatch
```

The updated job runs these exact commands sequentially:

```bash
python scripts/query_driven_paths.py \
  --edges graphs/spider/predicted_table_edges.csv \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --ranking graph --output results/query_graph_baseline.json

python scripts/query_driven_paths.py \
  --edges graphs/spider/predicted_table_edges.csv \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --ranking query-aware --query-weight 0.5 --length-penalty 0 \
  --output results/query_aware_paths.json
```

Keep the same CSV, threshold and search limits across ranking variants. Add
`--same-database` to **both** commands only for a separately labeled Spider
ablation. Cross-database traversal is the primary default. Changing K changes the
candidate set and must be treated separately from changing the scoring weights.
Record the repository revision, local diff if uncommitted, Python version,
CSV checksum (`sha256sum graphs/spider/predicted_table_edges.csv`), parameters and
Slurm logs. `search_truncated` must be reported, including empty-result queries.

## Existing graph-building pipeline (Jubail only)

Do not rerun this expensive pipeline merely to test reranking. If the current CSV
is missing but cached predictions exist, `sbatch export_edges.sbatch` is sufficient
once `logs/` exists. The exporter writes generated edge CSVs and uses a hardcoded
threshold matching the committed metadata; a future retraining requires checking
that threshold explicitly.

For a full reconstruction, the following is the existing script order. It requires
Spider SQLite files under `data/spider/databases`, the existing compatible ML
environment, and enough scheduled memory/time. Run preprocessing in an allocated
Jubail compute session, not locally or as an expensive login-node job:

```bash
cd /scratch/ds7491/graph_project
source .venv/bin/activate
mkdir -p logs results
python scripts/extract_spider_metadata.py
python scripts/build_spider_gold_graph.py
python scripts/build_spider_gold_hetero_graph.py
python scripts/build_candidate_features.py
python scripts/split_train_test_databases.py
python scripts/add_embedding_similarity.py
```

Then submit each existing stage **only after the preceding job succeeds**, checking
its output and error log before continuing:

```bash
sbatch build_cache.slurm
# After cache succeeds:
sbatch run_all_pairs.slurm
# After features succeed:
sbatch run_labels.slurm
# After labels succeed:
sbatch run_rf.slurm
# After training succeeds:
sbatch run_score_all.slurm
# After scoring succeeds:
sbatch export_edges.sbatch
# After export succeeds:
sbatch query_demo.sbatch
```

These are prerequisites and reproduction instructions, not jobs executed during
this change. There is no pinned dependency lockfile or embedding model revision
in the original project. Existing packages and cached model weights are needed
for faithful reconstruction; the standard-library reranker has no such dependency.

## Future evaluation plan (not implemented)

Build a question set with gold seed tables, required intermediate tables and
acceptable column-level join paths. Split questions by database/source to reduce
leakage. Validate source-independent joins manually or by execution. Preserve
alternative valid paths rather than assuming one unique gold route.

Compare graph-only, lexical mixture, optional length penalty, and same-database
ablation on identical candidates. Select alpha/lambda on validation data; freeze
them before held-out evaluation. Separately vary edge threshold, H, K and state
budget to measure the candidate-recall ceiling and computational trade-offs.

Report:

- Candidate path recall@K: fraction of questions with any acceptable path present
  before reranking; report separately from ranking quality.
- Intermediate-table precision/recall: intersection of selected versus required
  intermediate tables divided by selected/required counts; define empty-set
  conventions and macro-average across questions.
- Path hit@1 and MRR: first acceptable path rank, with reciprocal rank zero for
  no retrieved acceptable path; rank uses exact column joins when annotated.
- Join validity and execution success: valid selected hops and executable plans,
  distinguished from FK-label agreement.
- End-to-end answer accuracy after actual DBRAG integration.
- Runtime, peak memory, candidate count, truncation and no-path rates, and
  performance by hop count and within-/cross-source cases.

No numeric estimates for these future metrics are claimed here.

## 2026-10-28 extension: embeddings and independent LLM assessment

The implementation now compares four methods over identical graph candidates:
G (graph), graph+lexical, graph+embedding, and GPT assessment. The existing
`query_demo.sbatch` remains the offline two-baseline job. New
`query_compare.sbatch` adds local embeddings but does not enable GPT.

### Verification actually performed

`python3 -m unittest discover -s tests -v`: **31 tests pass** on local Python
3.12.1: the previous 15 plus 16 adapter/integration tests. Tests use controlled
vectors and mocked HTTP, with network transport denied by default in the adapter
suite. They cover cosine and score arithmetic, negative-cosine handling, zero
weight/ties, path serialization, encoder validation, offline loader arguments,
independent LLM evidence, strict response validation, abstention, refusal,
truncation, timeouts, HTTP errors, request-size limits, missing keys, empty paths,
comparison output and baseline preservation after failure.

For the original singer–stadium question and synthetic graph:

| Method/check | Observed test result | Evidence scope |
| --- | --- | --- |
| Graph | P1 archive before P2 concert bridge | Actual graph code on fixture |
| Lexical | P1 before P2, both coverage 2/3 | Actual lexical code on fixture |
| Embedding arithmetic | P2 before P1; S(P2)=0.8645, S(P1)=0.45125 at beta=0.5, lambda=0 | Controlled cosine values 1 and 0, **not MiniLM output** |
| LLM integration | P2 before P1; selection or NONE preserved | Mocked response, **not GPT inference** |

The embedding mock sets q=[1,0], concert-path v=[1,0], archive-path v=[0,1]. This
proves arithmetic and propagation only. It does not establish that a pretrained
model connects “performed” to “concert”. The LLM mock establishes validation and
failure behavior only. No improvement percentages, real-model relevance scores,
LLM accuracy or end-to-end QA gains have been measured.

SentenceTransformers, PyTorch and cached MiniLM weights were not available in the
local runtime. No dependency installation or model download was attempted. Live
OpenAI API testing was not authorized and no API calls were made. Official API
and library documentation were consulted via read-only web access.

### Offline semantic experiment on Jubail

Use a compatible embedding environment with a complete local model directory.
A concrete model directory varies by installation; replace the paths below with
your existing MiniLM export/snapshot and environment. No automatic model download
is performed by the reranker. Do not upgrade the RF training environment merely
to run this optional stage.

```bash
cd /scratch/ds7491/graph_project
mkdir -p logs results
export EMBEDDING_MODEL=/absolute/path/to/all-MiniLM-L6-v2
export EMBEDDING_VENV=/absolute/path/to/embedding-venv
sbatch query_compare.sbatch
```

Or, inside an allocated compute session with the embedding environment active:

```bash
python scripts/query_driven_paths.py \
  --edges graphs/spider/predicted_table_edges.csv \
  --question "Which singers performed at each stadium?" \
  --seed-a concert_singer.singer --seed-b concert_singer.stadium \
  --max-hops 5 --top-k 8 --max-states 150000 \
  --compare --ranking embedding --embedding-model "$EMBEDDING_MODEL" \
  --query-weight 0.5 --embedding-weight 0.5 --length-penalty 0 \
  --output results/query_comparison.json
```

Inspect `rankings` for the three offline rankings and `llm.status=not_requested`.
Record the CSV checksum, model snapshot/hash, `pip freeze`, Python/device details,
text version, all weights, candidate limits, metadata and runtime. Since model
identity is a filesystem path, the JSON alone does not prove file content identity.

### Later live GPT experiment — requires explicit approval before testing

With API use approved and `OPENAI_API_KEY` set securely, add `--llm` to the
comparison command above to compute all four rankings. For GPT alone, omit
`--compare` and use `--ranking llm --llm`; that run loads no embedding model.
The question, table names and joining columns are transmitted to OpenAI in this
mode. No data values or other reranker scores are sent.

Compare the four ordered ID lists on the same candidate set; use the separate
selected ID for abstention. Retain LLM failure/refusal cases in the evaluation
rather than dropping them. Record returned model, request hash, prompt version,
usage and whether a request failed. There is no automatic retry or automatic paid
Slurm job. LLM rankings may vary despite temperature zero.

### Remaining evaluation

The previously listed gold-path/bridge-table evaluation plan still applies. Add
independent graph+embedding and GPT baselines, tune alpha and beta separately on
validation questions, and freeze parameters before held-out testing. Compare
path hit@1/MRR and intermediate-table recall alongside candidate recall. Report
LLM abstention coverage and accuracy conditional on selection, plus unconditional
success rate, failure rate and token usage. Report similarity distributions before
assuming graph and embedding scores are calibrated. Test source-name and
presentation-order effects in the prompt. A combined embedding+GPT cascade is
future work, not part of this independent-method comparison.
