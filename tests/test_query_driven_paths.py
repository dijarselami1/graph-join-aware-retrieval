"""Offline regression tests: python3 -m unittest discover -s tests -v."""
import csv
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from scripts.query_driven_paths import (candidate_paths, lexical_tokens,
                                        question_relevance, rank_paths, read_graph)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/query_edges.csv'
A = 'concert_singer.singer'
B = 'concert_singer.stadium'


class QueryPathsTest(unittest.TestCase):
    def setUp(self):
        self.graph = read_graph(str(FIXTURE), 0)

    def paths(self, **kwargs):
        return candidate_paths(self.graph, A, B, **kwargs)[0]

    def test_bridge_recovery_and_graph_score(self):
        paths = rank_paths('', self.paths())
        self.assertEqual(len(paths), 2)
        bridge = paths[1]
        self.assertEqual(bridge['intermediate_tables'],
                         ['concert_singer.singer_in_concert', 'concert_singer.concert'])
        self.assertAlmostEqual(bridge['graph_score'], 0.9 ** 3)
        self.assertAlmostEqual(bridge['graph_log_score'], 3 * math.log(0.9))
        self.assertEqual(bridge['score_product'], round(0.9 ** 3, 8))
        self.assertEqual(paths[0]['original_graph_rank'], 1)

    def test_lexical_tokens(self):
        self.assertEqual(lexical_tokens('Which singers at each Stadium_ID?'),
                         {'singer', 'stadium', 'id'})
        self.assertEqual(lexical_tokens('SingerID singer-id'), {'singer', 'id'})
        self.assertEqual(lexical_tokens('the and which'), set())

    def test_relevance_and_column_evidence(self):
        path = {'tables': ['secret.places'], 'joins': [
            {'column_a': 'SingerID', 'column_b': 'venue_id'}]}
        result = question_relevance('singers secret', path)
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['matched_tokens'], ['singer'])
        self.assertEqual(question_relevance('', path)['score'], 0)
        self.assertEqual(question_relevance('unknown', path)['score'], 0)
        self.assertEqual(question_relevance('singers singers', path)['score'], 1)

    def test_reranking_and_formula(self):
        candidates = self.paths()
        ranked = rank_paths('Which singers performed at each concert stadium?', candidates,
                            query_weight=0.5, length_penalty=0.02, ranking='query-aware')
        self.assertEqual(ranked[0]['id'], 'P2')
        self.assertEqual(ranked[0]['original_graph_rank'], 2)
        self.assertEqual(ranked[0]['query_aware_rank'], 1)
        self.assertEqual(ranked[0]['question_relevance_score'], 0.75)
        expected = (0.5 * 0.9 ** 3 + 0.5 * 0.75) * math.exp(-0.04)
        self.assertAlmostEqual(ranked[0]['final_ranking_score'], expected)
        self.assertNotIn('query_aware_rank', candidates[0])
        penalized = rank_paths('Which singers performed at each concert stadium?',
                               candidates, length_penalty=0.1, ranking='query-aware')
        self.assertEqual(penalized[0]['id'], 'P1')

    def test_original_question_cannot_distinguish_candidates(self):
        ranked = rank_paths('Which singers performed at each stadium?', self.paths(),
                            ranking='query-aware')
        self.assertEqual([p['question_relevance_score'] for p in ranked], [2 / 3, 2 / 3])
        self.assertEqual(ranked[0]['id'], 'P1')

    def test_baseline_and_zero_weight(self):
        paths = self.paths()
        baseline = rank_paths('concert', paths, length_penalty=5)
        self.assertEqual([p['id'] for p in baseline], ['P1', 'P2'])
        self.assertEqual(baseline[0]['final_ranking_score'], baseline[0]['graph_score'])
        zero = rank_paths('concert', paths, query_weight=0, ranking='query-aware')
        self.assertEqual([p['id'] for p in zero], ['P1', 'P2'])
        tied = rank_paths('', paths, query_weight=1, ranking='query-aware')
        self.assertEqual([p['id'] for p in tied], ['P1', 'P2'])

    def test_cycles_and_hops(self):
        for path in self.paths(max_hops=10):
            self.assertEqual(len(path['tables']), len(set(path['tables'])))
        self.assertEqual(len(self.paths(max_hops=2)), 1)
        self.assertEqual(self.paths(max_hops=1), [])
        self.assertEqual(self.paths(max_hops=0), [])
        self.assertEqual(len(self.paths(k=1)), 1)

    def test_cross_database_and_optional_constraint(self):
        self.assertIn('unrelated.archive', self.paths()[0]['tables'])
        self.assertEqual(len(self.paths(same_database=True)), 1)
        self.assertEqual(len(candidate_paths(self.graph, A, 'unrelated.archive')[0]), 2)
        self.assertEqual(candidate_paths(self.graph, A, 'unrelated.archive',
                                         same_database=True), ([], False))

    def test_no_path_missing_and_same_seed(self):
        self.graph['isolated.table'] = []
        self.assertEqual(candidate_paths(self.graph, A, 'isolated.table'), ([], False))
        self.assertEqual(candidate_paths(self.graph, A, 'missing.table'), ([], False))
        same = candidate_paths(self.graph, A, A, max_hops=0)[0][0]
        self.assertEqual(same['tables'], [A])
        self.assertEqual(same['graph_score'], 1)

    def test_budget_flag(self):
        self.assertEqual(candidate_paths(self.graph, A, B, max_states=1), ([], True))
        self.assertFalse(candidate_paths(self.graph, A, A, k=1, max_states=1)[1])
        self.assertFalse(candidate_paths(self.graph, A, B, max_hops=0, max_states=1)[1])

    def test_reverse_join_columns(self):
        paths = candidate_paths(self.graph, B, A)[0]
        self.assertEqual(paths[0]['joins'][0]['column_a'], 'Stadium_ID')
        self.assertEqual(paths[0]['joins'][0]['column_b'], 'venue_id')

    def test_validation(self):
        for value in (-1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                rank_paths('', [], query_weight=value)
            with self.assertRaises(ValueError):
                read_graph(str(FIXTURE), value)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                rank_paths('', [], length_penalty=value)
        for kwargs in ({'max_hops': -1}, {'k': 0}, {'max_states': 0}):
            with self.assertRaises(ValueError):
                self.paths(**kwargs)

    def test_equal_parallel_joins_invalid_edges_and_row_order(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / 'edges.csv'
            header = ['db_a', 'table_a', 'column_a', 'db_b', 'table_b', 'column_b', 'score']
            rows = [['db', 'a', col, 'db', 'b', col, score] for col, score in
                    [('x', 1), ('y', 1), ('x', 1), ('z', 'nan'), ('z', 'inf'),
                     ('z', 0), ('z', -1), ('z', 1.1)]]
            reference = None
            for seed in range(5):
                random.Random(seed).shuffle(rows)
                with filename.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)
                paths = candidate_paths(read_graph(str(filename), 0), 'db.a', 'db.b')[0]
                self.assertEqual(len(paths), 3)
                if reference is None:
                    reference = paths
                self.assertEqual(paths, reference)

    def test_search_against_exhaustive_simple_paths(self):
        # Independent small-graph oracle checks ordering, scores, and completeness.
        expected = []
        middle = sorted(set(self.graph) - {A, B})
        for size in range(len(middle) + 1):
            for permutation in itertools.permutations(middle, size):
                nodes = [A] + list(permutation) + [B]
                product = 1.0
                for left, right in zip(nodes, nodes[1:]):
                    edges = [info for nxt, info in self.graph[left] if nxt == right]
                    if not edges:
                        break
                    product *= edges[0]['score']
                else:
                    expected.append((product, nodes))
        expected.sort(key=lambda item: (-item[0], item[1]))
        actual = self.paths(k=100)
        self.assertEqual([p['tables'] for p in actual], [nodes for _, nodes in expected])
        for path, (product, _) in zip(actual, expected):
            self.assertAlmostEqual(path['graph_score'], product)

    def test_cli_json_determinism_no_api_and_no_path(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'nested/result.json'
            command = [sys.executable, str(ROOT / 'scripts/query_driven_paths.py'),
                       '--edges', str(FIXTURE), '--question', 'concert singers stadium',
                       '--seed-a', A, '--seed-b', B, '--ranking', 'query-aware',
                       '--output', str(output)]
            subprocess.check_output(command)
            first = output.read_bytes()
            subprocess.check_output(command)
            self.assertEqual(first, output.read_bytes())
            data = json.loads(first)
            self.assertIsNone(data['llm_selection'])
            self.assertEqual(data['candidate_paths'][0]['id'], 'P2')
            for key in ('intermediate_tables', 'joins', 'graph_score',
                        'question_relevance_score', 'final_ranking_score',
                        'original_graph_rank', 'query_aware_rank'):
                self.assertIn(key, data['candidate_paths'][0])
            subprocess.check_output(command + ['--max-hops', '1'])
            self.assertEqual(json.loads(output.read_text())['candidate_paths'], [])
            subprocess.check_output(command + ['--seed-b', 'absent.table'])
            self.assertEqual(json.loads(output.read_text())['missing_seeds'], ['absent.table'])


if __name__ == '__main__':
    unittest.main()
