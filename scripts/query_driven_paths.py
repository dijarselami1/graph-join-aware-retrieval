#!/usr/bin/env python3
"""Prototype: question + seed tables -> candidate join paths -> optional LLM selection.
Python 3.6+; standard library only. Does not require DBRAG integration.
"""
import argparse
import csv
import heapq
import json
import math
import os
import sys
import urllib.request


def read_graph(filename, min_score):
    graph = {}
    with open(filename, newline='') as handle:
        for row in csv.DictReader(handle):
            s = float(row['score'])
            if s < min_score or s <= 0 or s > 1:
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
        adj.sort(key=lambda item: item[1]['score'], reverse=True)
    return graph


def candidate_paths(graph, start, end, max_hops=5, k=10, max_states=150000):
    # Enumerate simple paths in decreasing product-score order using a priority queue.
    # Stops after k results or a safety budget; not guaranteed exhaustive if budget reached.
    heap = [(0.0, (start,), ())]
    results = []
    popped = 0
    while heap and len(results) < k and popped < max_states:
        cost, nodes, joins = heapq.heappop(heap)
        popped += 1
        last = nodes[-1]
        if last == end:
            results.append({'id': 'P{}'.format(len(results) + 1),
                'tables': list(nodes), 'joins': list(joins),
                'score_product': round(math.exp(-cost), 8)})
            continue
        if len(nodes) - 1 >= max_hops:
            continue
        for nxt, info in graph.get(last, []):
            if nxt in nodes:
                continue
            step = {'table_a': last, 'column_a': info['left_column'],
                    'table_b': nxt, 'column_b': info['right_column'],
                    'joinability_score': round(info['score'], 6)}
            heapq.heappush(heap, (cost - math.log(info['score']),
                                   nodes + (nxt,), joins + (step,)))
    return results, popped >= max_states


def select_with_llm(question, candidates, model):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY is missing. Run without --llm to inspect candidates first.')
    candidate_json = json.dumps(candidates, indent=2)
    instructions = ('You are selecting join paths for multi-table question answering. '
      'The candidate edges are predictions, not proven valid joins. '
      'Use the question and shown table/column names to select the most relevant candidate. '
      'If insufficient evidence or none relevant, choose NONE. '
      'Do not invent columns, tables, or paths. Return a JSON object only with keys '
      'selected_path_id, reason, uncertainty. selected_path_id must be a candidate ID or NONE. '
      'Be brief and appropriately cautious.')
    body = json.dumps({'model': model, 'temperature': 0,
      'response_format': {'type': 'json_object'}, 'messages': [
      {'role': 'system', 'content': instructions},
      {'role': 'user', 'content': 'Question: '+question+'\nCandidate paths:\n'+candidate_json}]})
    request = urllib.request.Request('https://api.openai.com/v1/chat/completions',
      data=body.encode('utf-8'), headers={'Authorization': 'Bearer '+key,
      'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=50) as resp:
        answer = json.loads(resp.read().decode('utf-8'))
    obj = json.loads(answer['choices'][0]['message']['content'])
    valid = {x['id'] for x in candidates}
    if obj.get('selected_path_id') not in valid and obj.get('selected_path_id') != 'NONE':
        raise RuntimeError('LLM returned an unknown path ID: '+str(obj.get('selected_path_id')))
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', default='graphs/spider/predicted_table_edges.csv')
    p.add_argument('--question', required=True)
    p.add_argument('--seed-a', required=True, help='e.g. concert_singer.singer')
    p.add_argument('--seed-b', required=True, help='e.g. concert_singer.stadium')
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--max-hops', type=int, default=5)
    p.add_argument('--min-score', type=float, default=0.5160912733321336)
    p.add_argument('--llm', action='store_true', help='Call gpt-4o-mini to choose a path')
    p.add_argument('--output', default='results/query_path_demo.json')
    args = p.parse_args()
    graph = read_graph(args.edges, args.min_score)
    for node in [args.seed_a, args.seed_b]:
        if node not in graph:
            raise SystemExit('Seed not found in graph: '+node)
    paths, truncated = candidate_paths(graph,args.seed_a,args.seed_b,
                                       args.max_hops,args.top_k)
    data = {'question':args.question, 'seeds':[args.seed_a,args.seed_b],
            'candidate_paths':paths,'search_truncated':truncated,
            'llm_selection':None}
    if args.llm and paths:
        data['llm_selection'] = select_with_llm(args.question,paths,'gpt-4o-mini')
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(args.output,'w') as handle:
        json.dump(data,handle,indent=2)
    print('Found {} candidate paths{}'.format(len(paths),' (search budget reached)' if truncated else ''))
    for path in paths:
        print('{} score={:.4f}: {}'.format(path['id'],path['score_product'],
                                          ' -> '.join(path['tables'])))
    if data['llm_selection']:
        print('LLM selection:',json.dumps(data['llm_selection'],indent=2))
    print('Saved:',args.output)

if __name__=='__main__':
    main()
