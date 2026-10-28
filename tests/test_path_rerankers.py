"""Offline adapter tests: controlled embeddings and mocked HTTP, never model quality."""
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import urllib.error

from scripts import query_driven_paths as query
from scripts import path_rerankers as rerank

FIXTURE = Path(__file__).parent / 'fixtures/query_edges.csv'
QUESTION = 'Which singers performed at each stadium?'


class FakeEncoder:
    """Controlled vectors: question [1,0], archive orthogonal, concert aligned."""
    metadata = {'test_double': True}

    def encode(self, texts):
        self.texts = texts
        return [[1.0, 0.0]] + [[1.0, 0.0] if 'concert' in text else [0.0, 1.0]
                               for text in texts[1:]]


class RerankerTest(unittest.TestCase):
    def setUp(self):
        # Default-deny the only HTTP transport, even if a developer has an API key.
        self.network = patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        graph = query.read_graph(str(FIXTURE), 0)
        candidates, _ = query.candidate_paths(graph, 'concert_singer.singer', 'concert_singer.stadium')
        self.paths = query.rank_paths(QUESTION, candidates)

    def result(self, selected='P2'):
        return {'selected_path_id': selected, 'reason': 'Bridge connects the endpoints.',
                'uncertainty': 'Joins have not been executed.',
                'ranking': [{'path_id': pid, 'reason': 'Test assessment.',
                             'uncertainty': 'Unverified.'} for pid in ['P2', 'P1']]}

    def test_path_representation_excludes_scores_and_database_names(self):
        text = rerank.path_text(self.paths[1])
        self.assertIn('singer in concert', text)
        self.assertIn('concert.concert ID', text)
        self.assertNotIn('concert_singer', text)
        self.assertNotIn('0.9', text)
        self.assertNotIn('P2', text)
        self.assertEqual(text, rerank.path_text(copy.deepcopy(self.paths[1])))

    def test_cosine_validation(self):
        self.assertEqual(rerank.cosine_similarity([1, 0], [0, 2]), 0)
        self.assertEqual(rerank.cosine_similarity([2, 0], [-1, 0]), -1)
        for a, b in [([], []), ([0], [1]), ([1], [1, 2]), ([math.nan], [1]),
                     ([1], [math.inf])]:
            with self.assertRaises(ValueError):
                rerank.cosine_similarity(a, b)

    def test_embedding_rerank_formula_and_no_mutation(self):
        before = copy.deepcopy(self.paths)
        encoder = FakeEncoder()
        paths = rerank.rank_with_embeddings(QUESTION, self.paths, encoder, length_penalty=0.1)
        self.assertEqual(paths[0]['id'], 'P2')
        self.assertAlmostEqual(paths[0]['embedding_score'], (0.5 * 0.729 + 0.5) * math.exp(-0.2))
        self.assertEqual(paths[0]['embedding_rank'], 1)
        self.assertEqual(self.paths, before)
        self.assertEqual(encoder.texts[0], QUESTION)
        self.assertEqual(paths[0]['question_relevance_score'], 2 / 3)

    def test_negative_cosine_zero_weight_and_ties(self):
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [-1, 0], [0, 1]]
        paths = rerank.rank_with_embeddings(QUESTION, self.paths, encoder, weight=1)
        self.assertEqual([p['id'] for p in paths], ['P1', 'P2'])
        self.assertEqual(paths[0]['embedding_cosine'], -1)
        self.assertEqual(paths[0]['embedding_relevance_score'], 0)
        paths = rerank.rank_with_embeddings(QUESTION, self.paths, FakeEncoder(), weight=0)
        self.assertEqual([p['id'] for p in paths], ['P1', 'P2'])

    def test_embedding_empty_and_invalid_inputs(self):
        encoder = Mock()
        self.assertEqual(rerank.rank_with_embeddings('', [], encoder), [])
        encoder.encode.assert_not_called()
        for weight in [-0.1, 1.1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                rerank.rank_with_embeddings(QUESTION, self.paths, encoder, weight=weight)
        with self.assertRaises(ValueError):
            rerank.rank_with_embeddings(' ', self.paths, encoder)
        encoder.encode.return_value = [[1]]
        with self.assertRaises(ValueError):
            rerank.rank_with_embeddings(QUESTION, self.paths, encoder)

    def test_local_loader_offline_and_lazy(self):
        with tempfile.TemporaryDirectory() as temp:
            module = types.ModuleType('sentence_transformers')
            module.__version__ = 'test-double'
            module.SentenceTransformer = Mock()
            module.SentenceTransformer.return_value.max_seq_length = 256
            with patch.dict(sys.modules, {'sentence_transformers': module}):
                encoder = rerank.LocalSentenceEncoder(temp)
                module.SentenceTransformer.assert_called_once_with(
                    temp, device='cpu', local_files_only=True, trust_remote_code=False)
                encoder.encode(['hello'])
                self.assertEqual(encoder.metadata['max_seq_length'], 256)
            with patch.dict(sys.modules, {'sentence_transformers': None}):
                with self.assertRaises(RuntimeError):
                    rerank.LocalSentenceEncoder(temp)
        with self.assertRaises(ValueError):
            rerank.LocalSentenceEncoder('/missing/model')

    def test_llm_prompt_is_independent(self):
        embedded = rerank.rank_with_embeddings(QUESTION, self.paths, FakeEncoder())
        first = rerank.llm_request(QUESTION, self.paths)
        second = rerank.llm_request(QUESTION, embedded)
        self.assertEqual(first, second)
        evidence = json.loads(first['messages'][1]['content'])
        self.assertEqual(len(evidence['candidates']), 2)
        for candidate in evidence['candidates']:
            self.assertEqual(set(candidate), {'id', 'tables', 'joins'})
            for join in candidate['joins']:
                self.assertNotIn('joinability_score', join)
        self.assertEqual(first['model'], 'gpt-4o-mini')
        self.assertTrue(first['response_format']['json_schema']['strict'])

    def test_llm_validation_and_abstention(self):
        for selected in ['P2', 'NONE']:
            obj = self.result(selected)
            self.assertEqual(rerank.validate_llm_result(obj, self.paths), obj)
        bad = []
        obj = self.result('P1'); bad.append(obj)
        obj = self.result('unknown'); bad.append(obj)
        obj = self.result(); obj['ranking'].pop(); bad.append(obj)
        obj = self.result(); obj['ranking'][1]['path_id'] = 'P2'; bad.append(obj)
        obj = self.result(); obj['ranking'][1]['path_id'] = 'P3'; bad.append(obj)
        obj = self.result(); obj['reason'] = ''; bad.append(obj)
        obj = self.result(); obj['ranking'] = None; bad.append(obj)
        for obj in bad:
            with self.assertRaises(ValueError):
                rerank.validate_llm_result(obj, self.paths)

    def test_mocked_http_success(self):
        response = {'model': 'mock-gpt', 'usage': {'total_tokens': 123}, 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(self.result())}}]}
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), \
             patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result = rerank.select_with_llm(QUESTION, self.paths)
        self.assertEqual(result['selected_path_id'], 'P2')
        self.assertEqual(result['metadata']['usage']['total_tokens'], 123)
        self.assertEqual(len(result['metadata']['request_sha256']), 64)
        self.assertEqual(call.call_args[1]['timeout'], 50)
        self.assertEqual(json.loads(call.call_args[0][0].data)['model'], 'gpt-4o-mini')

    def test_missing_key_and_request_size_never_call_http(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                rerank.select_with_llm(QUESTION, self.paths)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}):
            with self.assertRaises(ValueError):
                rerank.select_with_llm('x' * 60001, self.paths)

    def test_mocked_http_failure_refusal_and_truncation(self):
        responses = [{'choices': [{'finish_reason': 'length', 'message': {}}]},
                     {'choices': [{'finish_reason': 'stop', 'message': {'refusal': 'No'}}]},
                     {'choices': [{'finish_reason': 'stop', 'message': {'content': '{bad'}}]},
                     {'choices': []}]
        for response in responses:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), \
                 patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(RuntimeError):
                    rerank.select_with_llm(QUESTION, self.paths)
        for error in [TimeoutError('timeout'), urllib.error.HTTPError('url', 429, 'rate limit', {}, None)]:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), \
                 patch('urllib.request.urlopen', side_effect=error) as call:
                with self.assertRaises(RuntimeError):
                    rerank.select_with_llm(QUESTION, self.paths)
                self.assertEqual(call.call_count, 1)

    def run_cli(self, output, extra):
        args = ['query_driven_paths.py', '--edges', str(FIXTURE), '--question', QUESTION,
                '--seed-a', 'concert_singer.singer', '--seed-b', 'concert_singer.stadium',
                '--output', str(output)] + extra
        with patch.object(sys, 'argv', args), patch('sys.stdout', new=io.StringIO()):
            query.main()

    def test_cli_compare_does_not_enable_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            with patch.object(query, 'LocalSentenceEncoder', return_value=FakeEncoder()), \
                 patch.object(query, 'select_with_llm', side_effect=AssertionError('LLM forbidden')):
                self.run_cli(output, ['--compare', '--ranking', 'embedding'])
            data = json.loads(output.read_text())
            self.assertEqual(data['schema_version'], 2)
            self.assertEqual(data['rankings']['graph']['path_ids'], ['P1', 'P2'])
            self.assertEqual(data['rankings']['lexical']['path_ids'], ['P1', 'P2'])
            self.assertEqual(data['rankings']['embedding']['path_ids'], ['P2', 'P1'])
            self.assertEqual(data['rankings']['llm']['status'], 'not_requested')
            self.assertEqual(data['candidate_paths'][0]['final_ranking_score'], 0.8645)

    def test_cli_all_four_rankings_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            with patch.object(query, 'LocalSentenceEncoder', return_value=FakeEncoder()), \
                 patch.object(query, 'select_with_llm', return_value=self.result()):
                self.run_cli(output, ['--compare', '--llm'])
                first = output.read_bytes()
                self.run_cli(output, ['--compare', '--llm'])
            self.assertEqual(first, output.read_bytes())
            data = json.loads(first)
            self.assertEqual(data['rankings']['llm']['path_ids'], ['P2', 'P1'])
            self.assertEqual(data['rankings']['embedding']['path_ids'], ['P2', 'P1'])
            self.assertEqual(data['candidate_paths'][0]['id'], 'P1')
            self.assertEqual(data['candidate_paths'][0]['final_ranking_score'], 0.9025)

    def test_cli_llm_without_embeddings_and_abstention(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            with patch.object(query, 'LocalSentenceEncoder', side_effect=AssertionError('No encoder')), \
                 patch.object(query, 'select_with_llm', return_value=self.result('NONE')):
                self.run_cli(output, ['--llm', '--ranking', 'llm'])
            data = json.loads(output.read_text())
            self.assertEqual(data['rankings']['embedding']['status'], 'not_requested')
            self.assertEqual(data['rankings']['llm']['selected_path_id'], 'NONE')
            self.assertEqual(data['candidate_paths'][0]['llm_rank'], 1)
            self.assertIsNone(data['candidate_paths'][0]['final_ranking_score'])

    def test_cli_llm_failure_preserves_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            with patch.object(query, 'select_with_llm', side_effect=RuntimeError('Test failure')), \
                 patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.run_cli(output, ['--llm', '--ranking', 'llm'])
            self.assertEqual(exc.exception.code, 2)
            data = json.loads(output.read_text())
            self.assertEqual(data['rankings']['llm']['status'], 'failed')
            self.assertIsNone(data['rankings']['llm']['path_ids'])
            self.assertEqual(data['output_order_status'], 'unavailable')
            self.assertEqual(len(data['candidate_paths']), 2)

    def test_cli_no_paths_skips_all_models(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results.json'
            with patch.object(query, 'LocalSentenceEncoder', side_effect=AssertionError('No encoder')), \
                 patch.object(query, 'select_with_llm', side_effect=AssertionError('No LLM')):
                self.run_cli(output, ['--compare', '--llm', '--max-hops', '0'])
            data = json.loads(output.read_text())
            self.assertEqual(data['candidate_paths'], [])
            self.assertEqual(data['rankings']['llm']['status'], 'no_candidates')


if __name__ == '__main__':
    unittest.main()
