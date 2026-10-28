# Development log

## 2026-10-28 — repository inspection

Inspected the repository inventory, all 26 Python scripts, Slurm/batch files,
committed logs, model metadata and static DOT demos before editing.
Also checked the committed graph/schema artifact structure during final verification. No README, docs directory or AGENTS.md was found in the
repository. The working tree was initially clean.

Existing functionality:

- SQLite metadata extraction and gold column/heterogeneous graph construction;
  diagnostic scripts distinguish unresolved FKs and duplicate undirected edges.
- Within-database structural features, database train/test split, MiniLM embeddings,
  structural rules, embedding thresholds and logistic-regression experiments.
- Global top-K structural candidates plus a separate all-pairs 11-feature RF
  pipeline. Cross-database labels are unknown, but global prediction retains them.
- Strongest-column projection into table edges, original single-best-path demo,
  and bounded priority-queue simple-path enumeration ranked by edge products.
- `query_driven_paths.py` already accepted a question but did not use it in ranking;
  it contained an opt-in remote LLM selection helper.
- Historical RF/count logs and static demos; no predicted CSV, actual model weights,
  local Spider data, README or automated offline regression suite.
- Most construction scripts and all batch jobs use `/scratch/ds7491/graph_project`.

## Changes implemented

| File | Change |
| --- | --- |
| `scripts/query_driven_paths.py` | Lexical token coverage, configurable reranking, score/rank audit fields, structured run metadata, optional database constraint, state-budget CLI, deterministic heap ties, finite-score checks and parameter validation |
| `query_demo.sbatch` | Paired graph-only/query-aware runs with distinct outputs; fail on command errors |
| `tests/test_query_driven_paths.py` | 15 offline tests including exhaustive-path oracle and CLI integration |
| `tests/fixtures/query_edges.csv` | Small synthetic bridge and cross-database distractor graph |
| `README.md` | Motivation, architecture, setup, local/Jubail usage and limitations |
| `docs/QUERY_AWARE_RETRIEVAL.md` | Traversal, pseudocode, lexical definition, equations, output contract and complexity |
| `docs/EXPERIMENTS.md` | Historical evidence, verified synthetic behavior, staged reproduction and future evaluation |
| `docs/DEVELOPMENT_LOG.md` | Inspection findings, decisions, verification and remaining work |

No trained models, datasets, large generated files, original graph construction
scripts or visualization repository were modified. No commits or pushes were made.

## Design decisions and compatibility

Kept the existing heap-based graph-only candidate search and graph ranking as the
default. Added reranking after generation rather than replacing the search.
`find_graph_path.py` remains unchanged. The old rounded `score_product` is retained;
new raw graph/log scores and unrounded edge scores permit auditing calculations.
Equal-score parallel joins previously risked dictionary comparisons in the heap;
new column signatures and serial numbers resolve that case deterministically.

The lexical baseline uses unique question-token coverage from table and joining
column names, excludes database names, and exposes matched tokens. It makes no
semantic or LLM claim. A fixed stopword list and deliberately simple plural rule
avoid dependencies but sacrifice linguistic quality. Seeds are included, which
can make all candidates tie on relevance; this limitation is explicitly tested.

The convex graph/relevance mixture uses alpha=0.5 as an untuned equal-weight
starting point. Optional exponential length penalty defaults to zero to avoid
further penalizing bridge paths already disadvantaged by graph products. Both
parameters are recorded. No normalization depends on the candidate set.

Cross-database traversal remains default; same-database filtering occurs only
within an explicitly requested search and never changes edge files. Missing seeds
now produce structured empty JSON rather than the former CLI exit, enabling future
UI consumers to distinguish missing seeds from disconnected ones. The budget flag
now reports actual budget-caused incompleteness instead of merely reaching a count.
Exact duplicate CSV rows remain distinct, preserving the existing multiedge input
behavior; the normal exporter provides one row per table pair.

The inherited LLM selector remains opt-in and separate. No new API dependency or
paid calls were introduced. New code/tests retain Python 3.6-compatible syntax
and standard-library APIs, but were executed only on the available local runtime.

## Verification and outcomes

`python3 -m unittest discover -s tests -v`: **15 tests passed**. The first test run
exposed an incorrect test expectation: lambda=0.1 outweighs the intended bridge's
lexical advantage. The reranking test now uses lambda=0.02, checks the exact
formula, and separately asserts that lambda=0.1 restores the shorter route.
This was a test-design correction, not a change to the scoring formula.

Tests verify intermediate recovery, products, tokenization, column evidence,
reranking and its limitations, cycle/hop/K/budget handling, optional database
filtering, empty results, same-seed behavior, reverse columns, validation,
equal-score/duplicate edges, shuffled-input determinism, exhaustive enumeration,
and repeatable CLI JSON. Historical RF metrics are only transcribed from existing
logs. Full Spider retrieval and Jubail runtime execution remain unverified locally.

Additional checks passed: the original script loaded from Git and the updated
search return identical table paths and rounded products on the synthetic fixture;
`bash -n query_demo.sbatch`; `git diff --check`; and parsing both new Python sources
with Python 3.6 grammar. Execution used Python 3.12.1, not an actual 3.6 interpreter.
Read-only XML inspection found 4,497 nodes/793 edges in the gold column GraphML
and 5,370 nodes/5,292 edges in the heterogeneous GraphML. These are artifact counts,
not new model or retrieval results.

## Remaining work

Run the paired Jubail job against the actual predicted edge CSV. Preserve input
checksums, revision and run parameters. Annotate question-level paths and evaluate
candidate recall separately from ranking; tune weights on validation only. Improve
linguistic matching, retain alternative/composite joins, measure memory on dense
graphs, validate cross-source joins, and integrate with DBRAG and visualization in
future work. No held-out path-ranking or QA improvement is claimed.

## 2026-10-28 — semantic embeddings and independent LLM extension

Inspected the current query script, tests, all four documentation files, batch
job, RF metadata and existing MiniLM column embedding script. The prior task's
uncommitted changes were retained. Explained the design before editing. Consulted
official SentenceTransformer and OpenAI structured-output documentation without
calling model APIs. Local inspection found no SentenceTransformers/PyTorch install
or MiniLM cache.

Implemented:

- New `scripts/path_rerankers.py`: optional offline SentenceTransformer loader,
  explicit table/join path texts, validated cosine arithmetic, independent
  graph+embedding mixture, strict GPT-4o-mini request/response adapter and method
  comparison serialization.
- Extended `scripts/query_driven_paths.py`: embedding/LLM/lexical output modes,
  opt-in comparison, separate embedding weight/model/device arguments, schema v2
  with optional-method statuses, baseline retention and exit 2 on LLM failure.
  Removed the older single-choice helper in favor of the validated ranking helper;
  the public imported `select_with_llm` name remains available.
- New `query_compare.sbatch`: offline three-method comparison using an existing
  local model and optional separate embedding venv. Existing baseline batch job
  is preserved; no job enables paid LLM calls automatically.
- New `tests/test_path_rerankers.py`: 16 tests with fake encoder and mocked HTTP.
- Updated README and retrieval/experiment/development documentation to distinguish
  implemented adapters, synthetic checks, live experiments and proposed work.

The RF graph, graph traversal, lexical scoring and cross-database behavior were
not changed in this extension. Optional dependencies load only in embedding mode.
The core uses Python 3.6-compatible syntax, but a compatible modern ML environment
may be needed for embeddings; actual Python 3.6 and Jubail execution remain
unverified. No models, datasets or visualization code were edited.

Scoring decisions: clip negative cosine to zero positive evidence, preserve raw
cosine, combine with G using a separately configurable untuned beta=0.5, and keep
the existing length penalty. No lexical score is mixed into embeddings. GPT gets
names/joins without other methods' scores, returns a full ranking and may abstain.
No scalar GPT confidence is fabricated. Model/prompt metadata and statuses support
future controlled experiments; they do not imply model quality has been measured.

Validation: all **31 offline tests pass** on Python 3.12.1. Tests cover the old
baselines plus adapter arithmetic, isolation, strict validation, failure paths and
four-method JSON. Real MiniLM inference and live GPT remain unverified. No network
model calls, model downloads, dependency installation, commits or pushes occurred.
Remaining work: provision or identify the existing compatible Jubail model/env,
run real offline semantic comparison, obtain approval before live API testing,
then annotate/evaluate held-out queries and tune weights. Semantic caching,
execution validation and combined semantic+LLM cascades remain proposed work.
