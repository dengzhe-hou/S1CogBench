"""Offline integration tests. All callbacks and source rows are synthetic fixtures, not model results."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1cogbench import cli, results, worker
from s1cogbench.modes import BY_KEY
from s1cogbench.planning import describe, inspect_adapter


class ToyAgent:
    last_native_choice = 'A'

    def __init__(self, p):
        self.p, self.states = p, []

    def choice_probabilities(self, state, question, keys):
        self.states.append(state)
        return self.p


def fixture_run(base, rule, predictive=False, probabilities=(0., 1., 0., 0.), ids=(7,)):
    agent, updates = ToyAgent(probabilities), []
    data = {sid: [{'user_key': 3, 'ground_key': 2, 'step': i + 1} for i in range(3)] for sid in ids}

    class Dialogue:
        def __init__(self, family, mode):
            self.history = []

        def prompt(self, row):
            return tuple(self.history)

        def update(self, row, key, letter):
            self.history.append(key)
            updates.append((key, letter))

    source = SimpleNamespace(LETTERS='ABCD', QUESTION={}, test_data=lambda: data, Dialogue=Dialogue,
                             task_metrics=lambda rows, keys: {'accuracy': sum(k == 2 for k in keys) / len(keys)})
    scorer = SimpleNamespace(predictive_metrics=lambda rows, _: {'human_nll_legal': 1.0})
    ctx = SimpleNamespace(agent=lambda: agent, action_rule=rule, smoke=False,
                          check=lambda count, expected, label: None)
    mode = BY_KEY['wcst:human_predictive' if predictive else 'wcst:autonomous_generative']
    base.mkdir()
    with patch.object(worker, 'module', side_effect=lambda name: source if name == 'src.wcst_task' else scorer):
        report = worker.wcst(mode, base, ctx)
    records = [json.loads(line) for line in (base / 'decisions.jsonl').read_text().splitlines()]
    return report, records, agent.states, updates


class EvaluationEntryTests(unittest.TestCase):
    def test_plan_never_imports_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'agent.py'
            p.write_text("raise RuntimeError('adapter import must not happen')\ndef create_agent(device):\n    return None\n")
            run = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'plan', '--adapter', str(p), '--json'],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report['selected_modes'], 27)
            self.assertFalse(report['adapter']['imported'])
            self.assertEqual(report['model_API_calls'], 0)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_sample_executes_distribution_and_advances_its_own_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, rows, states, updates = fixture_run(Path(tmp) / 'sample', 'sample')
            self.assertEqual([r['key'] for r in rows], [2, 2, 2])
            self.assertEqual([r['native_choice'] for r in rows], ['A', 'A', 'A'])
            self.assertEqual([r['executed_choice'] for r in rows], ['B', 'B', 'B'])
            self.assertEqual(states, [(), (2,), (2, 2)])
            self.assertEqual(updates, [(2, 'B')] * 3)
            self.assertEqual(report['summary']['executed_action_rule'], 'sample')

    def test_native_and_predictive_history_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, native, states, updates = fixture_run(Path(tmp) / 'native', 'native')
            self.assertEqual([r['key'] for r in native], [1, 1, 1])
            self.assertEqual(states, [(), (1,), (1, 1)])
            _, predicted, states, updates = fixture_run(Path(tmp) / 'prediction', 'sample', predictive=True)
            self.assertEqual([r['key'] for r in predicted], [1, 1, 1])
            self.assertEqual(states, [(), (3,), (3, 3)])
            self.assertEqual(updates, [(3, None)] * 3)

    def test_sample_seed_uses_original_subject_id(self):
        import numpy as np
        p = [.3, .7, 0., 0.]
        with tempfile.TemporaryDirectory() as tmp:
            _, rows, _, _ = fixture_run(Path(tmp) / 'first', 'sample', probabilities=p, ids=(7, 9))
            _, repeated, _, _ = fixture_run(Path(tmp) / 'second', 'sample', probabilities=p, ids=(9, 7))
            for sid in (7, 9):
                expected = np.random.RandomState(10000 + sid).choice(list('ABCD'), size=3, p=p).tolist()
                self.assertEqual([r['executed_choice'] for r in rows if r['subject_id'] == sid], expected)
                self.assertEqual([r['executed_choice'] for r in repeated if r['subject_id'] == sid], expected)

    def test_unsupported_sampling_stops_before_import(self):
        args = SimpleNamespace(tasks=['nonstationary_bandit'], action_rule='sample')
        with patch.object(cli, 'adapter_module') as imported:
            with self.assertRaisesRegex(SystemExit, 'not implemented'):
                cli.cmd_eval(args)
            imported.assert_not_called()

    def test_text_parser_placeholders_cannot_be_sampled(self):
        args = SimpleNamespace(tasks=['wcst:autonomous_generative'], action_rule='sample',
                               adapter=Path('text_adapter.py'), smoke=False)
        with patch.object(cli, 'adapter_module', return_value=SimpleNamespace(PROBABILITY_READOUT=False)):
            with self.assertRaisesRegex(SystemExit, 'actual legal-option probabilities'):
                cli.cmd_eval(args)

    def test_sampled_rows_cannot_share_native_comparison_context(self):
        mode = BY_KEY['wcst:autonomous_generative']
        record = {'status': 'complete', 'primary': {'value': .5, 'value_status': 'finite', 'n': 75}}
        native = results.row(mode, {**record, 'action_rule': 'native'}, 'test-fixture')
        sampled = results.row(mode, {**record, 'action_rule': 'sample'}, 'test-fixture')
        self.assertNotEqual(native['protocol'], sampled['protocol'])
        self.assertEqual(native['protocol'], mode.primary['protocol'])
        self.assertEqual(sampled['runtime_context']['effective_action_rule'], 'sample')
        rows = describe([mode], 'sample')
        self.assertNotIn('last_native_choice', rows[0]['adapter_interface'])
        bandit = describe([BY_KEY['stationary_multiarmed_bandit']], 'native')[0]
        self.assertIn('last_native_choice', bandit['adapter_interface'])

    def test_published_templates_have_the_documented_factory(self):
        for name in ['hf_causal_lm_choice.py', 'decision_api_choice.py', 'generative_api_choice.py']:
            report = inspect_adapter(ROOT / 'examples/templates' / name)
            self.assertFalse(report['agent_created'])


if __name__ == '__main__':
    unittest.main(verbosity=2)