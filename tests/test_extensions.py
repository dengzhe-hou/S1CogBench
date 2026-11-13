"""Offline tests of the extension track. Every task and value here is a synthetic fixture, not a model result."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1cogbench import community, extensions, results, worker
from s1cogbench.modes import MODES
from s1cogbench.planning import comparison_context, describe, effective_action_rule

FIXTURES = ROOT / 'tests/fixtures/extensions'
UNIFORM = ROOT / 'examples/uniform_reference_agent.py'
MODES_X, MODULES_X = extensions.discover(FIXTURES)
BY_KEY_X = {m.key: m for m in MODES_X}


def registered():
    """Make the fixtures the installed extension tasks inside a with block."""
    stack = contextlib.ExitStack()
    for p in (patch.dict(extensions.ALL_BY_KEY, BY_KEY_X), patch.dict(extensions.EXTENSION_BY_KEY, BY_KEY_X),
              patch.dict(extensions.MODULES, MODULES_X), patch.object(extensions, 'EXTENSION_MODES', MODES_X)):
        stack.enter_context(p)
    return stack


class Agent:
    """Fixed probabilities for every request."""
    def __init__(self, probabilities=None, report=0.5, native=None):
        self.probabilities, self.report, self.native = probabilities, report, native

    def choice_probabilities(self, state, question, keys):
        if self.native is not None:
            self.last_native_choice = self.native
        return self.probabilities or [1 / len(keys)] * len(keys)

    def posterior_probability(self, state, question):
        return self.report


class Context:
    adapter, device, action_rule, label, smoke, limit, trace = UNIFORM, 'cpu', 'native', 'fixture', False, 0, []

    def __init__(self, agent):
        self._agent = agent

    def begin(self, mode):
        pass

    def agent(self):
        return self._agent

    def check(self, count, expected, what):
        if count != expected:
            raise ValueError(f'{what}: {count} != expected {expected}')


def write_task(directory, name, body):
    path = Path(directory) / name / 'task.py'
    path.parent.mkdir(parents=True)
    path.write_text(body)


class ExtensionTests(unittest.TestCase):
    def test_paper_suite_is_unchanged(self):
        self.assertEqual([m.task for m in MODES_X], ['toy_choice', 'toy_environment', 'toy_report'])
        self.assertTrue(all(m.track == 'extension' for m in MODES_X))
        self.assertTrue(all(m.track == 'core' for m in MODES))
        with registered():
            self.assertEqual(extensions.select(['all']), list(MODES))
            self.assertEqual(len(extensions.select(['all'])), 27)
            self.assertEqual([m.key for m in extensions.select(['extensions'])], [m.key for m in MODES_X])
            chosen = extensions.select(['rlwm', 'toy_choice'])
            self.assertEqual([m.key for m in chosen], ['rlwm', 'toy_choice'])
            self.assertEqual(describe(chosen, 'native')[1]['track'], 'extension')

    def test_task_files_are_validated(self):
        good = (FIXTURES / 'toy_choice/task.py').read_text()
        cases = {'rlwm': good.replace("'toy_choice'", "'rlwm'"),
                 'other_name': good,
                 'no_reference': good.replace("'reference': {", "'unused': {"),
                 'no_loader': good.replace('def load_trials', 'def unused'),
                 'odd_metric': good.replace("'metric': 'human_nll'", "'metric': 'reward'")}
        for name, body in cases.items():
            with tempfile.TemporaryDirectory() as tmp, self.subTest(name):
                write_task(tmp, name, body)
                with self.assertRaises(ValueError):
                    extensions.discover(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            write_task(tmp, '_template', 'raise SystemExit("templates are not loaded")')
            self.assertEqual(extensions.discover(tmp), ([], {}))

    def test_builtin_metrics(self):
        trials = MODULES_X['toy_choice'].load_trials(ROOT)
        s = extensions.score_trials('human_nll', Agent(), trials, io.StringIO())
        self.assertAlmostEqual(s['primary_value'], math.log(3))
        self.assertEqual((s['primary_n'], s['primary_value_is_infinite']), (4, False))
        s = extensions.score_trials('human_nll', Agent([1., 0., 0.]), trials, io.StringIO())
        self.assertEqual((s['primary_value'], s['primary_value_is_infinite']), (None, True))
        accuracy = [dict(t, correct=t['observed']) for t in trials]
        s = extensions.score_trials('accuracy', Agent([.2, .5, .3]), accuracy, io.StringIO())
        self.assertEqual(s['primary_value'], .25)  # argmax B matches one of four
        s = extensions.score_trials('accuracy', Agent(native='A'), accuracy, io.StringIO())
        self.assertEqual(s['primary_value'], .5)   # a native choice takes precedence over argmax
        reports = MODULES_X['toy_report'].load_trials(ROOT)
        s = extensions.score_trials('probability_mae', Agent(), reports, io.StringIO())
        self.assertAlmostEqual(s['primary_value'], (.4 + 0 + .3) / 3)
        s = extensions.score_trials('probability_mae', Agent(report=None), reports, io.StringIO())
        self.assertEqual((s['primary_value'], s['invalid_responses'], s['responses']), (None, 3, 3))
        with self.assertRaises(ValueError):
            extensions.score_trials('human_nll', Agent([.5, .6, 0.]), trials, io.StringIO())

    def test_autonomous_tasks_keep_their_action_rule(self):
        mode = BY_KEY_X['toy_environment']
        self.assertEqual(effective_action_rule(mode, 'sample'), 'sample')
        self.assertEqual(comparison_context(mode, 'sample')['protocol'], 'toy_two_arm_sample')
        self.assertEqual(effective_action_rule(BY_KEY_X['toy_choice'], 'sample'), 'not_applied')

    def test_run_package_and_board(self):
        with registered(), tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            rows, keys = [], ['toy_choice', 'toy_environment', 'toy_report']
            for key in keys:
                mode = BY_KEY_X[key]
                out = run / mode.task / mode.mode
                out.mkdir(parents=True)
                result = worker.run_mode(mode, out, Context(Agent(native='B')))
                self.assertEqual(result['status'], 'complete', result.get('error'))
                rows.append(results.row(mode, result, 'fixture-model'))
            self.assertAlmostEqual(rows[0]['value'], math.log(3))
            self.assertEqual(rows[1]['value'], 1.0)
            files = {'run_config.json': {'smoke': False, 'modes': keys, 's1cogbench_version': 'test',
                                         'action_rule': 'native', 'probability_readout': True},
                     'summary.json': {'modes': 3, 'complete': 3}, 'results.json': {'rows': rows}}
            metadata = {name: 'Synthetic fixture, not measured' for name in community.METADATA}
            metadata.update(model_name='fixture-model', evidence_url='https://example.org/fixture')
            files['metadata.json'] = metadata
            for name, value in files.items():
                (run / name).write_text(json.dumps(value))
            item = community.prepare(run, run / 'metadata.json', 'fixture-model')
            site = Path(tmp) / 'site'
            (site / 'results').mkdir(parents=True)
            (site / 'results/latest').symlink_to(ROOT / 'results/latest')
            (site / 'submissions').mkdir()
            (site / 'submissions/fixture-model.json').write_text(json.dumps(item))
            board = community.build_board(site)
            tracks = [b['track'] for b in board['boards']]
            self.assertEqual(tracks, sorted(tracks, key=lambda t: t != 'core'))
            extension = {b['mode_key']: b for b in board['boards'] if b['track'] == 'extension'}
            self.assertEqual(set(extension), set(keys))
            self.assertEqual(extension['toy_choice']['direction'], 'lower')
            self.assertEqual(extension['toy_environment']['entries'][0]['rank'], 1)
            relabeled = json.loads(json.dumps(item))
            relabeled['rows'][0]['protocol'] = 'rlwm_protocol'
            with self.assertRaises(ValueError):
                community.validate_submission(relabeled)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'the CLI worker imports torch')
    def test_cli_eval_of_extension_tasks(self):
        out = ROOT / 'runs' / f'_test_extensions_{os.getpid()}'
        env = dict(os.environ, S1COGBENCH_EXTENSIONS=str(FIXTURES), PYTHONDONTWRITEBYTECODE='1')
        try:
            done = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'eval', '--adapter', str(UNIFORM),
                                   '--tasks', 'extensions', '--output', str(out)],
                                  cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            rows = {r['task']: r for r in json.loads((out / 'results.json').read_text())['rows']}
            self.assertAlmostEqual(rows['toy_choice']['value'], math.log(3))
            self.assertAlmostEqual(rows['toy_report']['value'], .7 / 3)
            self.assertEqual(rows['toy_environment']['value'], 0.0)  # the uniform adapter's native choice is A
            listed = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'list'], cwd=ROOT, env=env,
                                    capture_output=True, text=True, check=True).stdout
            self.assertIn('Extension tasks (3)', listed)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
