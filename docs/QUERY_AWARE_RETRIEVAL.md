# Query-aware path selection

## Implemented design

`scripts/query_driven_paths.py` retains two stages: graph-only candidate generation,
then separate optional lexical, embedding or LLM assessment. Graph search never sees the question. It accepts
an undirected predicted edge CSV with header:

```text
db_a,table_a,column_a,db_b,table_b,column_b,score
```

Nodes use case-sensitive `database.table` identifiers (database identifiers must
not contain dots). CSV scores must parse as numbers. Nonfinite scores and values
outside `(0,1]` are skipped, as are scores below `--min-score` and table self-edges.
Malformed rows produce a CLI error. The default threshold matches committed RF
metadata; this is an inherited edge threshold, not a tuned relevance parameter.
Reverse adjacency swaps column endpoints. Input files are never modified.

The normal exporter supplies one strongest column edge per table pair. If a custom
CSV contains parallel joins, they remain separate candidates, including exact
repeated rows. Candidate IDs identify join sequences, not unique table sequences.
Duplicate rows can consume candidate slots; deduplicate custom inputs if unwanted.

## Traversal, step by step

1. Put the seed-only path in a min-heap at cost zero.
2. Pop the partial path with smallest cumulative negative log score.
3. If it reaches the destination, record it without extending it.
4. Otherwise, if it already uses `max_hops` edges, stop extending it.
5. For each adjacent edge, skip any neighbor already in this path. There is no
   global visited set: alternative paths through the same table remain possible.
6. If `same_database` is enabled, require each neighbor to share the source seed's
   database. Different-database seed pairs then return no candidates.
7. Push each eligible extension with cost increased by `-log(edge_score)`.
8. Stop at `top_k` complete paths, empty heap, or `max_states` pops.

Because scores are in `(0,1]`, costs are nonnegative. Completed paths are emitted
in nonincreasing product order. A state-budget cutoff can return fewer than K
paths; `search_truncated` is true only when work remains, fewer than K paths were
found, and the budget caused termination. False does not mean all possible paths
were enumerated: K or the hop limit can still exclude paths.

Heap ties compare the table sequence, then the column-pair sequence, then a
monotonic serial number. This preserves the old table-sequence tie ordering and
fixes dictionary-comparison errors for equal-score parallel joins. CSV adjacency
is sorted by descending score, neighbor, and column names for repeatability.
Query-score ties preserve original graph rank. Floating-point arithmetic still
limits exact equivalence across runtimes and mathematically equal products.

Absent seeds yield empty candidates and `missing_seeds` in JSON. Disconnected
seeds yield empty candidates without missing seeds. Identical existing seeds
produce a zero-hop path with empty joins and product 1. Hop limit may be zero;
K and state budget must be positive. No-path is a valid successful CLI result.

## Lexical relevance

`lexical_tokens` splits snake case, punctuation, camel case and acronym boundaries,
casefolds Unicode text, removes a fixed English stopword list, and strips a trailing
`s` from tokens longer than three characters unless they end in `ss`, `us`, or
`is`. The exact stopwords are:

```text
a an the which what who where when how at in on of for to from
and or is are was were be been each all by with do does did
```

This maps `singers` to `singer`, but is not a linguistic lemmatizer: `cities` becomes
`citie`; synonyms, tense, negation and intent are not understood. Numbers are kept.
Let Q be unique question tokens and T(P) the union of tokens from every table name
(including seeds, excluding database prefixes) and both joining columns of each
hop. Then:

```text
R(Q,P) = |Q intersect T(P)| / |Q|, if |Q| > 0; otherwise 0.
```

This fixed-denominator coverage lies in [0,1], counts repeated words once, and is
independent of which other candidates are retrieved. No IDF corpus or min/max
normalization is fitted. Token unions do not penalize irrelevant vocabulary, so
long paths can gain coverage accidentally. Seed tokens can dominate. The output
exposes sorted query, path and matched tokens for inspection.

## Scoring equations and choices

For path P with h edges and edge predictions s_i:

```text
G(P) = product_i s_i = exp(-sum_i(-log(s_i)))
L(P) = exp(-lambda * max(0, h-1))
S(Q,P) = ((1-alpha)*G(P) + alpha*R(Q,P)) * L(P)
```

`--query-weight alpha` must be finite and in [0,1]. Default 0.5 is an **untuned
equal mixture**, not an empirically justified optimum. Both terms have fixed
[0,1] bounds; this does not imply comparable calibration or distributions. The
additive mixture allows a path with zero lexical overlap to retain graph evidence.
Alpha 0 and lambda 0 recover graph ranking; alpha 1 uses lexical coverage alone
before the optional length penalty.

`--length-penalty lambda` must be finite and nonnegative. Default zero disables
it because G already tends to decrease with length. Each hop beyond the first
multiplies the mixture by exp(-lambda); lambda=log(2) would halve it per extra hop.
Zero/one-hop paths receive factor 1. Strong penalties can suppress necessary
bridge tables; treat this as an ablation and tune on validation questions only.

`--ranking graph` sorts by the original search rank and uses G as the final score,
regardless of mixture settings. `--ranking query-aware` sorts the same candidates
by S. Both modes calculate and export both ranks. Scores use unrounded floats;
legacy `score_product` remains rounded to eight decimals for compatibility.
`graph_log_score` retains the negative-log-search-derived log product. Very long
paths can underflow the displayed product; graph search order still uses log costs.
Neither RF scores nor their products establish calibrated joint probabilities or
independence of joins.

## Pseudocode

```text
candidates = []
heap = [(cost=0, path=[seed_a], joins=[])]
while heap nonempty and len(candidates) < K and pops < budget:
    cost, path, joins = pop_min(heap)
    pops += 1
    if last(path) == seed_b:
        emit candidate with G=exp(-cost) and graph rank
        continue
    if edge_count(path) == max_hops: continue
    for next, edge in adjacency[last(path)]:
        if next in path or excluded_by_optional_database_constraint: continue
        push(heap, cost-log(edge.score), path+[next], joins+[oriented(edge)])
for candidate in candidates:
    R = lexical_question_coverage(candidate)
    S = ((1-alpha)*G + alpha*R) * exp(-lambda*extra_hops)
assign query ranks by descending S, breaking ties with graph rank
export in selected ranking order
```

## Singer–stadium example and intermediate recovery

For “Which singers performed at each stadium?” with seeds
`concert_singer.singer` and `concert_singer.stadium`, the expected bridge is:

```text
singer --Singer_ID/Singer_ID--> singer_in_concert
       --concert_ID/concert_ID--> concert
       --Stadium_ID/Stadium_ID--> stadium
```

The graph discovers `singer_in_concert` and `concert` without requiring them as
seeds. Reranking does not discover new tables; it scores paths already found.
In the synthetic fixture, each bridge edge scores 0.9, so G=0.729. An unrelated
cross-database archive route has two edges of 0.95, so G=0.9025. These are invented
**test inputs**, not measured Spider predictions.

| Synthetic question | Bridge R | Archive R | Graph rank of bridge | Query rank of bridge |
| --- | ---: | ---: | ---: | ---: |
| Which singers performed at each stadium? | 2/3 | 2/3 | 2 | 2 |
| Which singers performed at each concert stadium? | 3/4 | 2/4 | 2 | 1 |

The table uses alpha=0.5, lambda=0. For the second question, bridge S=0.7395 and
archive S=0.70125. With lambda=0.1 the shorter archive wins again. The extra word
“concert” deliberately supplies distinguishable lexical evidence. The baseline
cannot infer that “performed” implies a concert. This example demonstrates
mechanics, not improved accuracy on real questions.

## Baseline JSON fields (retained in schema_version 2)

Top-level fields: `question`, `seeds` (source, destination), `parameters` (input
path, threshold, hop/candidate/state limits, constraint, selected ranking, weights,
lexical method version), `candidate_paths`, `search_truncated`, `missing_seeds`,
and the retained optional `llm_selection` (null in offline runs).

Each candidate has:

| Field | Meaning |
| --- | --- |
| `id` | P1, P2, ... in graph discovery order, unchanged by reranking |
| `tables`, `intermediate_tables` | Ordered full path and its interior |
| `joins` | Ordered hops with table_a, column_a, table_b, column_b, joinability_score |
| `hop_count` | Number of edges |
| `score_product` | Legacy rounded graph product |
| `graph_score`, `graph_log_score` | Unrounded product and its log |
| `question_relevance_score`, `lexical_evidence` | Coverage and token audit trail |
| `length_penalty_factor`, `query_aware_score` | L and S |
| `final_ranking_score` | G or S according to selected mode |
| `original_graph_rank`, `query_aware_rank` | One-based ranks within this candidate set |

IDs are run-local; use tables and joins for cross-run comparison. No timestamps
are emitted, allowing identical inputs/parameters to produce identical JSON bytes
on the same runtime for offline graph/lexical runs. Embedding numerics can vary
across runtimes, and GPT output is not guaranteed deterministic. Parameters record
the CSV path, not its checksum; archive
input checksums and the source revision separately for research reproduction.
This structure is ready for future visualization consumption; no UI is integrated.

## Complexity and failure modes

For V nodes and E retained undirected edges, loading costs O(E) memory and up to
O(E log E) total adjacency sorting time. Let H be max hops, Delta maximum degree,
B popped states and M peak queued states. Each expansion examines up to Delta
neighbors, tests path membership and copies tuples of length O(H). A conservative
bound is O(B*Delta*H*log(M+1)) time and O(E+M*H+K*H) memory, including heap comparison
costs. Simple-path counts grow combinatorially; the pop budget does not bound heap
memory directly. If L is total lexical input length across candidates, reranking
costs O(L + K log K), plus token-audit sorting, and stores the audit evidence.

Candidate recall limits reranking recall. Thresholding, strongest-join projection,
hop limits, K and state limits can each hide the correct route. Spurious bridges,
common IDs, ambiguous names, absent schema descriptions, simplistic tokenization,
incorrect seeds, missing edges and domain shift are unresolved. Two-endpoint simple
paths cannot express branching join plans or repeated-table/self-join aliases.
Cross-database traversal is intentionally supported; it is not evidence that
those joins are valid.

## Proposed future work (not implemented)

Automatic seed retrieval, learned weight calibration, execution-based join
validation, composite joins, branching plans, DBRAG integration, and interactive
visualization. Semantic and LLM adapters are now implemented below; evaluating
their real retrieval quality remains future work.

## Semantic embedding branch (implemented)

`scripts/path_rerankers.py` provides optional adapters; importing it loads no ML
packages. `LocalSentenceEncoder` loads an existing SentenceTransformer directory
on CPU by default, using `local_files_only=True` and `trust_remote_code=False`.
There is no automatic fallback to lexical matching when loading fails. The loader
requires an installed SentenceTransformers version supporting these arguments;
older Jubail environments can still run the core baselines unchanged.

`path_text` serializes each path as ordered table names and explicit joins:

```text
Tables: singer -> singer in concert -> concert -> stadium. Proposed joins:
singer.Singer ID = singer in concert.Singer ID;
singer in concert.concert ID = concert.concert ID;
concert.Stadium ID = stadium.Stadium ID.
```

This is formatting of existing names, not generated descriptions. Camel/snake case
is split without the lexical branch's stemming or stopwords. Database prefixes
and scores are omitted from embeddings. The text version is `tables_and_joins_v1`.
All path texts and the question are encoded together, in batches of 32. Metadata
records the model directory, library version, device and maximum sequence length.
The model can truncate long inputs; original serialized text is saved but no
claim is made that all of it influenced the embedding. Cached column embeddings
are not reused as question/path embeddings; this is independent of the RF feature.

For query vector q and path vector v:

```text
C(Q,P) = dot(q,v) / (norm(q)*norm(v))
E(Q,P) = max(0, C(Q,P))
S_embedding(Q,P) = ((1-beta)*G(P) + beta*E(Q,P)) * L(P)
```

Vectors must be finite, nonempty, same-dimensional and have nonzero norms.
Floating-point cosine is clamped to [-1,1]. Negative cosine is retained in output
but contributes zero positive relevance; this fixed transform avoids interpreting
orthogonal vectors as 0.5 relevance, as an affine [-1,1] mapping would do. It is a
heuristic, not calibrated probability. `--embedding-weight beta` is finite in
[0,1], default 0.5, and independent of `--query-weight alpha`. L uses the existing
length penalty. Beta=0 and lambda=0 preserve graph ordering. Ties use graph rank.
No lexical feature enters this score and no training is performed.

Encoder inference dominates runtime. Beyond inference, cosine costs O(K*d), text
construction is linear in serialized names, and sorting costs O(K log K). The
loaded model and embedding arrays add memory beyond the graph search. There is
no persistent embedding cache yet. Real-model determinism depends on model files,
package versions, device and floating-point kernels.

## Independent GPT-4o-mini branch (implemented, not live-tested)

`--llm` makes one Chat Completions request with strict JSON schema, temperature 0,
a 4096-token output cap and 50-second timeout. Requests above 60000 serialized
bytes are rejected before transport. There are no automatic retries. This byte
cap is a practical guard, not a token/context-length calculation; larger candidate
sets may still exceed the output cap and are rejected if incomplete.

The prompt receives the question, run-local IDs, full table identifiers and
oriented column joins. It does **not** receive graph scores, lexical scores,
embedding scores, prior ranks or other methods' selections. Candidates use stable
table/column order independently of output mode. IDs still originate in graph
order, so the prompt explicitly says IDs and order are not relevance evidence;
position/identifier bias is not eliminated and should be studied in evaluation.
Full database-qualified names prevent table-identity ambiguity, but can introduce
source-name bias. Cross-database joins remain allowed.

The prompt asks for a brief assessment and uncertainty for every path, a full
ranking and either the first-ranked path or `NONE`. It treats schema/question text
as data, instructs against invented joins, and distinguishes proposed joins from
proven validity. This is LLM-assisted assessment based on available names, not
SQL execution, guaranteed reasoning correctness, or hidden chain-of-thought output.

The validator requires every candidate exactly once and rejects unknown IDs,
duplicates, omitted paths, blank assessments and inconsistent selection. Refusals,
incomplete output, malformed JSON and HTTP errors produce failure, not a made-up
ranking. `NONE` is valid abstention even with a full relative ranking. When the
LLM fails, the CLI saves baseline results and the failure status, then exits 2.
If LLM ordering was requested, `output_order_status=unavailable`; the stored list
remains in graph order with null final scores. No model is called for empty paths.

The metadata stores requested/returned model, temperature, prompt version,
request SHA-256 and API token usage if returned. It does not store credentials.
Temperature zero does not guarantee identical live output. The supplied
`gpt-4o-mini` alias is preserved; record returned model identity for experiments.
The existing single-choice helper contract is extended to require full rankings.
There is no numeric LLM relevance score or embedding+LLM blend.

## Comparison and schema_version 2

The baseline fields remain, including `query_aware_score/rank` for lexical results.
New CLI modes are `lexical` (alias for `query-aware`), `embedding`, and `llm`.
`--compare` computes embeddings along with both baseline ranks; it does not change
output order unless `--ranking` is supplied and never enables LLM calls.
`--ranking llm` requires `--llm`. LLM-only evaluation needs no embedding package.

New fields:

| Field | Meaning |
| --- | --- |
| `embedding_text` | Exact serialized path input |
| `embedding_cosine` | Raw signed cosine |
| `embedding_relevance_score` | Positive cosine E |
| `embedding_score`, `embedding_rank` | Independent graph+embedding mixture and rank |
| `llm_rank`, `llm_assessment` | Validated rank and brief reason/uncertainty |
| `embedding_metadata` | Loaded encoder identity/configuration, or null |
| `llm_selection` | Full assessment ranking, selected ID/NONE, reason, uncertainty and metadata |
| `rankings` | graph/lexical/embedding/llm ordered IDs and statuses |
| `output_order_status` | Whether the requested output ordering is available |

Unavailable optional per-path fields are omitted. Unrequested method `path_ids`
are null, while a computed empty list is `[]`. LLM statuses distinguish
`not_requested`, `no_candidates`, `ok`, `failed`; embedding is `no_candidates`
when requested but skipped on an empty candidate set. An embedding dependency or
inference failure aborts before output rather than silently claiming a comparison.
Graph and lexical statuses are `ok` even for a valid empty search.

`final_ranking_score` is G, lexical S, or embedding S according to output mode;
it is null in LLM mode because a rank is not a calibrated scalar score. Always
check `llm_selection.selected_path_id` for abstention rather than treating the
first array element as the chosen path. Both optional methods use the *same*
graph candidates; neither can recover a path excluded during generation.

## Implemented versus proposed

Implemented: local model adapter, explicit path text, cosine/graph mixture,
independent structured GPT assessment, rank comparison, validation and offline
tests. Proposed: real-model quality evaluation, learned weights, prompt-order
ablations, semantic caching, controlled embedding+LLM cascades and execution-based
validation. Four independent rankings are not an end-to-end QA evaluation.

References checked during implementation:
[SentenceTransformer loading and encoding](https://sbert.net/docs/package_reference/sentence_transformer/model.html),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
