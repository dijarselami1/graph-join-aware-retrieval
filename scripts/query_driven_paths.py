#!/usr/bin/env python3
"""Graph-only candidate generation with separate lexical, embedding and LLM rankings.
Core: Python 3.6+ standard library; embeddings require optional sentence-transformers. Does not require DBRAG integration.
"""
import argparse
import csv
import heapq
import json
import math
import itertools
import re
import os
import sys

if __package__:
    from .path_rerankers import (LocalSentenceEncoder, rank_with_embeddings,
                                 select_with_llm, ranking_comparison)
else:
    from path_rerankers import (LocalSentenceEncoder, rank_with_embeddings,
                                select_with_llm, ranking_comparison)


def read_graph(filename, min_score):
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be finite and between 0 and 1")
    graph = {}
    with open(filename, newline='') as handle:
        for row in csv.DictReader(handle):
            s = float(row['score'])
            if not math.isfinite(s) or s < min_score or s <= 0 or s > 1:
                continue
            a = row['db_a'] + '.' + row['table_a']
            b = row['db_b'] + '.' + row['table_b']
            if a == b:
                continue
            edge = {'score': s, 'left_column': row['column_a'],
                    'right_column': row['column_b']}
            graph.setdefault(a, []).append((b, edge))
            graph.setdefault(b, []).append((a, {'score': s,
                'left_column': row['column_b'], 'right_column': row['column_a']}))
    for adj in graph.values():
        adj.sort(key=lambda item: (-item[1]['score'], item[0],
                                  item[1]['left_column'], item[1]['right_column']))
    return graph


def candidate_paths(graph, start, end, max_hops=5, k=10, max_states=150000,
                    same_database=False):
    # Enumerate simple paths in decreasing product-score order using a priority queue.
    # Stops after k results or a safety budget; not guaranteed exhaustive if budget reached.
    if max_hops < 0 or k < 1 or max_states < 1:
        raise ValueError("max_hops must be nonnegative; k and max_states must be positive")
    if start not in graph or end not in graph:
        return [], False
    if same_database and start.split('.', 1)[0] != end.split('.', 1)[0]:
        return [], False
    serial = itertools.count()
    # Preserve the historical table-sequence tie order; never compare join dictionaries.
    heap = [(0.0, (start,), (), next(serial), ())]
    results = []
    popped = 0
    while heap and len(results) < k and popped < max_states:
        cost, nodes, signature, _, joins = heapq.heappop(heap)
        popped += 1
        last = nodes[-1]
        if last == end:
            results.append({'id': 'P{}'.format(len(results) + 1),
                'tables': list(nodes), 'joins': list(joins),
                'score_product': round(math.exp(-cost), 8),
                'graph_score': math.exp(-cost),
                'graph_log_score': -cost})
            continue
        if len(nodes) - 1 >= max_hops:
            continue
        for nxt, info in graph.get(last, []):
            if same_database and nxt.split('.', 1)[0] != start.split('.', 1)[0]:
                continue
            if nxt in nodes:
                continue
            step = {'table_a': last, 'column_a': info['left_column'],
                    'table_b': nxt, 'column_b': info['right_column'],
                    'joinability_score': info['score']}
            heapq.heappush(heap, (cost - math.log(info['score']),
                                   nodes + (nxt,),
                                   signature + ((info['left_column'], info['right_column']),),
                                   next(serial), joins + (step,)))
    return results, bool(heap and len(results) < k and popped >= max_states)


# Fixed, deliberately small English stopword list, versioned with the baseline.
STOPWORDS = frozenset("a an the which what who where when how at in on of for to from "
                      "and or is are was were be been each all by with do does did".split())


def lexical_tokens(text):
    """Split identifiers, casefold, remove stopwords and apply a simple plural rule."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = set()
    for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith('s') and not token.endswith(('ss', 'us', 'is')):
            token = token[:-1]
        tokens.add(token)
    return tokens


def question_relevance(question, path):
    """Unique query-token coverage; database identifiers are not lexical evidence."""
    query = lexical_tokens(question)
    evidence = set()
    for table in path['tables']:
        evidence.update(lexical_tokens(table.split('.', 1)[-1]))
    for join in path['joins']:
        evidence.update(lexical_tokens(join['column_a']))
        evidence.update(lexical_tokens(join['column_b']))
    matched = query & evidence
    return {'score': len(matched) / len(query) if query else 0.0,
            'query_tokens': sorted(query), 'path_tokens': sorted(evidence),
            'matched_tokens': sorted(matched)}


def rank_paths(question, candidates, query_weight=0.5, length_penalty=0.0,
               ranking='graph'):
    """Annotate both ranks without changing candidate generation or mutating inputs."""
    if not math.isfinite(query_weight) or not 0 <= query_weight <= 1:
        raise ValueError('query_weight must be finite and between 0 and 1')
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError('length_penalty must be finite and nonnegative')
    if ranking not in ('graph', 'query-aware'):
        raise ValueError('ranking must be graph or query-aware')
    annotated = []
    for rank, candidate in enumerate(candidates, 1):
        path = dict(candidate)
        relevance = question_relevance(question, path)
        hops = len(path['joins'])
        penalty = math.exp(-length_penalty * max(0, hops - 1))
        graph_score = path['graph_score']
        query_score = ((1 - query_weight) * graph_score +
                       query_weight * relevance['score']) * penalty
        path.update(intermediate_tables=path['tables'][1:-1], hop_count=hops,
                    question_relevance_score=relevance['score'], lexical_evidence=relevance,
                    length_penalty_factor=penalty, query_aware_score=query_score,
                    original_graph_rank=rank,
                    final_ranking_score=graph_score if ranking == 'graph' else query_score)
        annotated.append(path)
    reranked = sorted(annotated, key=lambda x: (-x['query_aware_score'], x['original_graph_rank']))
    for rank, path in enumerate(reranked, 1):
        path['query_aware_rank'] = rank
    return annotated if ranking == 'graph' else reranked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', default='graphs/spider/predicted_table_edges.csv')
    p.add_argument('--question', required=True)
    p.add_argument('--seed-a', required=True, help='e.g. concert_singer.singer')
    p.add_argument('--seed-b', required=True, help='e.g. concert_singer.stadium')
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--max-hops', type=int, default=5)
    p.add_argument('--min-score', type=float, default=0.5160912733321336)
    p.add_argument('--ranking', choices=['graph', 'query-aware', 'lexical', 'embedding', 'llm'], default='graph')
    p.add_argument('--query-weight', type=float, default=0.5,
                   help='Convex mixture weight in [0,1]; untuned default 0.5')
    p.add_argument('--length-penalty', type=float, default=0.0,
                   help='Nonnegative exponential penalty per hop beyond the first')
    p.add_argument('--max-states', type=int, default=150000)
    p.add_argument('--same-database', action='store_true',
                   help='Spider ablation only; restrict traversal to the seed database')
    p.add_argument('--compare', action='store_true',
                   help='Compute graph, lexical and embedding ranks; LLM still requires --llm')
    p.add_argument('--embedding-model', help='Existing local SentenceTransformer model directory')
    p.add_argument('--embedding-device', default='cpu')
    p.add_argument('--embedding-weight', type=float, default=0.5)
    p.add_argument('--llm', action='store_true', help='Explicitly enable a paid gpt-4o-mini ranking/selection request')
    p.add_argument('--output', default='results/query_path_demo.json')
    args = p.parse_args()
    if args.ranking == 'llm' and not args.llm:
        p.error('--ranking llm requires explicit --llm')
    if not math.isfinite(args.embedding_weight) or not 0 <= args.embedding_weight <= 1:
        p.error('--embedding-weight must be finite and between 0 and 1')
    use_embeddings = args.compare or args.ranking == 'embedding'
    embedding_metadata = None
    try:
        graph = read_graph(args.edges, args.min_score)
        paths, truncated = candidate_paths(graph, args.seed_a, args.seed_b,
                                           args.max_hops, args.top_k, args.max_states,
                                           args.same_database)
        paths = rank_paths(args.question, paths, args.query_weight,
                           args.length_penalty, 'graph')
        if use_embeddings and paths:
            encoder = LocalSentenceEncoder(args.embedding_model, args.embedding_device)
            paths = rank_with_embeddings(args.question, paths, encoder,
                                         args.embedding_weight, args.length_penalty)
            embedding_metadata = encoder.metadata
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        p.error(str(exc))
    data = {'schema_version': 2, 'question': args.question,
            'seeds': [args.seed_a, args.seed_b], 'candidate_paths': paths,
            'search_truncated': truncated,
            'missing_seeds': [s for s in (args.seed_a, args.seed_b) if s not in graph],
            'parameters': {'edges': args.edges, 'max_hops': args.max_hops,
                           'top_k': args.top_k, 'min_score': args.min_score,
                           'max_states': args.max_states, 'same_database': args.same_database,
                           'ranking': args.ranking, 'query_weight': args.query_weight,
                           'length_penalty': args.length_penalty,
                           'lexical_method': 'query_token_coverage_v1',
                           'compare': args.compare, 'embedding_weight': args.embedding_weight,
                           'embedding_model': args.embedding_model,
                           'embedding_device': args.embedding_device,
                           'llm_requested': args.llm},
            'embedding_metadata': embedding_metadata,
            'llm_selection': None}
    llm_status = 'not_requested'
    if args.llm:
        llm_status = 'no_candidates'
        if paths:
            try:
                data['llm_selection'] = select_with_llm(args.question, paths, 'gpt-4o-mini')
                llm_status = 'ok'
            except (RuntimeError, ValueError) as exc:
                llm_status = 'failed'
                data['llm_error'] = str(exc)
    if data['llm_selection']:
        assessments = {row['path_id']: (rank, row) for rank, row in
                       enumerate(data['llm_selection']['ranking'], 1)}
        for path in paths:
            path['llm_rank'], path['llm_assessment'] = assessments[path['id']]
    data['rankings'] = ranking_comparison(paths, use_embeddings, data['llm_selection'], llm_status)
    rank_field = {'graph': 'original_graph_rank', 'lexical': 'query_aware_rank',
                  'query-aware': 'query_aware_rank', 'embedding': 'embedding_rank',
                  'llm': 'llm_rank'}[args.ranking]
    score_field = {'graph': 'graph_score', 'lexical': 'query_aware_score',
                   'query-aware': 'query_aware_score', 'embedding': 'embedding_score'}.get(args.ranking)
    paths.sort(key=lambda path: path.get(rank_field, path['original_graph_rank']))
    for path in paths:
        path['final_ranking_score'] = path[score_field] if score_field else None
    data['output_order_status'] = 'unavailable' if args.ranking == 'llm' and llm_status == 'failed' else 'ok'

    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(args.output,'w') as handle:
        json.dump(data,handle,indent=2,allow_nan=False)
    print('Found {} candidate paths{}'.format(len(paths),' (search budget reached)' if truncated else ''))
    for path in paths:
        print('{} graph={:.4f} final={}: {}'.format(path['id'],path['graph_score'],
                                          path['final_ranking_score'],
                                          ' -> '.join(path['tables'])))
    if args.compare or args.llm:
        print('Rankings:', json.dumps(data['rankings'], sort_keys=True))
    if data['llm_selection']:
        print('LLM selection:',json.dumps(data['llm_selection'],indent=2))
    print('Saved:',args.output)
    if llm_status == 'failed':
        print('LLM failed; baseline results saved: ' + data['llm_error'], file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':
    main()
