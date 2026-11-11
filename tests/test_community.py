"""Offline submission fixtures. None of these values are model measurements."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1cogbench import community, results
from s1cogbench.modes import BY_KEY


def fixture(key='bayesian_urn_inference', rule='native', value=.2, status='finite', n=None):
    mode = BY_KEY[key]
    n = mode.scored_expected if n is None else n
    record = {'status': 'complete', 'inputs': mode.inputs_expected, 'scored': mode.scored_expected,
              'action_rule': rule, 'primary': {'value': value, 'value_status': status, 'n': n}}
    row = results.row(mode, record, 'fixture-model')
    metadata = {name: 'Synthetic fixture, not measured' for name in community.METADATA}
    metadata.update(model_name='fixture-model', evidence_url='https://example.org/fixture')
    return {'format_version': 1, 'benchmark': community.BENCHMARK, 'submission_id': 'fixture-model',
            'metadata': metadata,
            'run': {'s1cogbench_version': '0.2.0rc1', 'action_rule': rule, 'probability_readout': True},
            'coverage': [{'mode_key': key, 'status': 'complete', 'inputs': mode.inputs_expected,
                          'scored': mode.scored_expected, 'primary_n': n}], 'rows': [row]}, record


def save_run(path, item, record):
    mode = BY_KEY[item['coverage'][0]['mode_key']]
    path.mkdir(parents=True)
    (path / mode.task / mode.mode).mkdir(parents=True)
    files = {'metadata.json': item['metadata'],
             'run_config.json': {**item['run'], 'smoke': False, 'modes': [mode.key]},
             'summary.json': {'modes': 1, 'complete': 1},
             'results.json': {'rows': item['rows']},
             f'{mode.task}/{mode.mode}/mode_result.json': record}
    for name, value in files.items():
        (path / name).write_text(json.dumps(value))


class CommunityTests(unittest.TestCase):
    def test_package_and_check_cli_do_not_import_model_or_copy_trials(self):
        item, record = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run, submissions = Path(tmp) / 'run', Path(tmp) / 'submissions'
            save_run(run, item, record)
            (run / 'private_trials.jsonl').write_text('RESTRICTED_FIXTURE_MUST_STAY_LOCAL')
            output = submissions / 'fixture-model.json'
            proc = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'package',
                                   '--run', str(run), '--metadata', str(run / 'metadata.json'),
                                   '--id', 'fixture-model', '--output', str(output)],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn('RESTRICTED_FIXTURE', output.read_text())
            self.assertEqual(list(submissions.iterdir()), [output])
            checked = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'check-submissions',
                                      '--directory', str(submissions)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(community.read_json(output)['rows'], item['rows'])

    def test_smoke_and_partial_runs_are_not_packaged(self):
        item, record = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            save_run(run, item, record)
            path = run / 'run_config.json'
            config = community.read_json(path)
            config['smoke'] = True
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'smoke'):
                community.prepare(run, run / 'metadata.json', 'fixture-model')
            config['smoke'] = False
            path.write_text(json.dumps(config))
            (run / 'summary.json').write_text(json.dumps({'modes': 1, 'complete': 0}))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                community.prepare(run, run / 'metadata.json', 'fixture-model')

    def test_saved_score_change_is_detected(self):
        item, record = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            save_run(run, item, record)
            changed = copy.deepcopy(item['rows'])
            changed[0]['value'] = .1
            (run / 'results.json').write_text(json.dumps({'rows': changed}))
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                community.prepare(run, run / 'metadata.json', 'fixture-model')

    def test_participant_denominator_is_not_action_count(self):
        item, _ = fixture('wcst:autonomous_generative', n=75)
        self.assertEqual(community.validate_submission(item)['rows'][0]['n'], 75)
        item['rows'][0]['n'] = 12754
        with self.assertRaisesRegex(ValueError, 'denominator'):
            community.validate_submission(item)

    def test_incomplete_counts_and_wrong_protocol_are_rejected(self):
        item, _ = fixture()
        item['coverage'][0]['scored'] -= 1
        with self.assertRaisesRegex(ValueError, 'counts'):
            community.validate_submission(item)
        item, _ = fixture()
        item['rows'][0]['protocol'] = 'different-prompt-protocol'
        with self.assertRaisesRegex(ValueError, 'protocol'):
            community.validate_submission(item)

    def test_native_and_sampling_remain_in_separate_boards(self):
        native, _ = fixture('wcst:autonomous_generative', n=75)
        sampled, _ = fixture('wcst:autonomous_generative', rule='sample', n=75)
        for item in (native, sampled):
            community.validate_submission(item)
        with patch.object(community, 'load_submissions', return_value=[native, sampled]):
            board = community.build_board()
        contexts = [b['context']['protocol'] for b in board['boards']
                    if any(e.get('origin') == 'community' for e in b['entries'])]
        self.assertEqual(len(set(contexts)), 2)

    def test_infinity_undefined_and_negative_reward_are_preserved(self):
        entries = []
        for name, value, status in [('finite', 2., 'finite'), ('infinite', 'Infinity', 'defined_positive_infinity'),
                                    ('undefined', None, 'undefined')]:
            entries.append({'model': name, 'value': value, 'value_status': status})
        ranked = community.rank_entries(entries, 'lower')
        self.assertEqual([e['rank'] for e in ranked], [1, 2, None])
        self.assertEqual(ranked[1]['value'], 'Infinity')
        negative = [{'model': 'a', 'value': -5., 'value_status': 'finite'},
                    {'model': 'b', 'value': -2., 'value_status': 'finite'}]
        self.assertEqual(community.rank_entries(negative, 'higher')[0]['model'], 'b')

    def test_ties_and_preference_have_no_invented_order(self):
        entries = [{'model': name, 'value': value, 'value_status': 'finite'}
                   for name, value in [('a', .2), ('b', .2), ('c', .3)]]
        self.assertEqual([e['rank'] for e in community.rank_entries(entries, 'lower')], [1, 1, 3])
        self.assertTrue(all(e['rank'] is None for e in community.rank_entries(entries, '')))

    def test_inconsistent_value_and_missing_metadata_fail(self):
        item, _ = fixture()
        item['rows'][0]['value'] = 'Infinity'
        with self.assertRaisesRegex(ValueError, 'value_status'):
            community.validate_submission(item)
        item, _ = fixture()
        item['metadata']['checkpoint'] = 'REPLACE_WITH_MODEL'
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            community.validate_submission(item)

    def test_publication_preserves_paper_data_and_has_no_fixture_entries(self):
        paths = [ROOT / 'results/aggregate/results.json', ROOT / 'results/aggregate/table2.json']
        original = [p.read_bytes() for p in paths]
        board = community.build_board()
        self.assertEqual([p.read_bytes() for p in paths], original)
        self.assertGreaterEqual(len(board['boards']), 27)
        self.assertEqual(board['submission_count'], len(community.load_submissions(ROOT / 'submissions')))
        self.assertFalse(any(e['model'] == 'fixture-model' for b in board['boards'] for e in b['entries']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
