#!/usr/bin/env python3
"""Inspect one labeled path-ranking JSON without loading a model or making API calls.

This is a per-question diagnostic, not a benchmark. Gold paths are supplied by
the researcher and should be checked at the column-join level before publication.
"""
import argparse
import json
import math


def evaluate(data, gold_ids):
    paths = data['candidate_paths']
    by_id = {path['id']: path for path in paths}
    if len(by_id) != len(paths):
        raise ValueError('Duplicate candidate IDs in result')
    unknown = set(gold_ids) - set(by_id)
    if unknown:
        raise ValueError('Gold candidate IDs absent: ' + ', '.join(sorted(unknown)))
    if not gold_ids:
        raise ValueError('At least one verified gold candidate ID is required')
    orders = data['rankings']
    summary = {}
    for method in ('graph', 'lexical', 'embedding', 'llm'):
        info = orders[method]
        if info['status'] != 'ok':
            summary[method] = {'status': info['status'], 'best_gold_rank': None,
                               'hit_at_1': None, 'reciprocal_rank': None}
            continue
        ids = info['path_ids']
        if len(ids) != len(by_id) or set(ids) != set(by_id):
            raise ValueError(method + ' ranking is not a permutation of candidates')
        rank = min(ids.index(gold) + 1 for gold in gold_ids)
        summary[method] = {'status': 'ok', 'best_gold_rank': rank,
                           'hit_at_1': rank == 1, 'reciprocal_rank': 1.0 / rank}
    ablation = data.get('embedding_ablation')
    if ablation:
        for label, key in (('embedding_selected_text', 'selected_path_ids'),
                           ('embedding_alternate_text', 'alternate_path_ids')):
            ids = ablation[key]
            if len(ids) != len(by_id) or set(ids) != set(by_id):
                raise ValueError('Ablation ranking is not a permutation')
            rank = min(ids.index(gold) + 1 for gold in gold_ids)
            summary[label] = {'status': 'ok', 'text_version': ablation[
                'selected_text_version' if key == 'selected_path_ids' else 'alternate_text_version'],
                'best_gold_rank': rank, 'hit_at_1': rank == 1,
                'reciprocal_rank': 1.0 / rank}
    distractors = []
    if orders['embedding']['status'] == 'ok':
        highest_gold = max(by_id[gold]['embedding_cosine'] for gold in gold_ids)
        for path in paths:
            if path['id'] in gold_ids:
                continue
            cosine = path['embedding_cosine']
            if not math.isfinite(cosine):
                raise ValueError('Nonfinite embedding cosine')
            if cosine > highest_gold:
                distractors.append({'id': path['id'], 'cosine': cosine,
                                    'graph_score': path['graph_score'],
                                    'embedding_rank': path['embedding_rank'],
                                    'tables': path['tables']})
        distractors.sort(key=lambda x: (-x['cosine'], x['id']))
    return {'question': data['question'], 'gold_ids': sorted(gold_ids),
            'candidate_count': len(paths), 'search_truncated': data.get('search_truncated'),
            'methods': summary, 'higher_cosine_distractors': distractors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Existing query_driven_paths JSON')
    parser.add_argument('--gold-id', action='append', required=True,
                        help='Verified acceptable candidate ID; repeat for multiple valid paths')
    args = parser.parse_args()
    with open(args.result) as stream:
        data = json.load(stream)
    print(json.dumps(evaluate(data, set(args.gold_id)), indent=2))


if __name__ == '__main__':
    main()
