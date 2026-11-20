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
                               adapter=cli.adapter_spec('text_adapter.py'), smoke=False, model_label=None)
        with patch.object(cli, 'adapter_module', return_value=SimpleNamespace(PROBABILITY_READOUT=False)):
            with self.assertRaisesRegex(SystemExit, 'actual legal-option probabilities'):
                cli.cmd_eval(args)

    def test_hf_adapter_spec_uses_the_template(self):
        spec = cli.adapter_spec('hf:Qwen/Qwen3.5-4B@abc123')
        self.assertEqual(spec.path, cli.HF_TEMPLATE)
        self.assertEqual(spec.env, {'S1_HF_MODEL': 'Qwen/Qwen3.5-4B', 'S1_HF_REVISION': 'abc123'})
        self.assertEqual(spec.label, 'Qwen3.5-4B')
        self.assertEqual(cli.adapter_spec('hf:/models/my-llama/').env, {'S1_HF_MODEL': '/models/my-llama/'})
        self.assertEqual(cli.adapter_spec('hf:/models/my-llama/').label, 'my-llama')
        plain = cli.adapter_spec('my_agent.py')
        self.assertEqual((plain.path, plain.env, plain.label), (Path('my_agent.py'), {}, None))
        with self.assertRaises(cli.argparse.ArgumentTypeError):
            cli.adapter_spec('hf:')

    def test_plan_accepts_hf_adapter_without_importing_it(self):
        out = subprocess.run([sys.executable, '-B', '-m', 's1cogbench', 'plan', '--adapter', 'hf:org/model',
                              '--tasks', 'lite', '--json'], cwd=ROOT, capture_output=True, text=True, check=True)
        report = json.loads(out.stdout)
        self.assertEqual(report['adapter']['spec'], 'hf:org/model')
        self.assertEqual(Path(report['adapter']['file']), cli.HF_TEMPLATE)
        self.assertFalse(report['adapter']['imported'])
        self.assertEqual(report['selected_modes'], 8)

    def test_eval_passes_the_hf_model_to_workers(self):
        name = 'runs/.test-hf-adapter'
        out = ROOT / name
        self.addCleanup(lambda: __import__('shutil').rmtree(out, ignore_errors=True))
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append((command, env))
            return SimpleNamespace(returncode=0)
        runnable = lambda modes, root: {'python': '3', 'packages': {}, 'modes': [
            {'key': m.key, 'label': m.label, 'runnable': True, 'reasons': []} for m in modes]}
        with patch.object(cli, 'adapter_module', return_value=SimpleNamespace(PROBABILITY_READOUT=True)), \
                patch.object(cli, 'hf_model_problem', return_value=None), \
                patch.object(cli, 'diagnose', side_effect=runnable), \
                patch.object(cli.subprocess, 'run', side_effect=run), patch.object(cli, 'finish'), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(['eval', '--adapter', 'hf:org/tiny-model', '--tasks', 'false_belief_story_qa', '--output', name])
        config = json.loads((out / 'run_config.json').read_text())
        self.assertEqual(config['adapter_spec'], 'hf:org/tiny-model')
        self.assertEqual(Path(config['adapter']), cli.HF_TEMPLATE.resolve())
        self.assertEqual(config['model_label'], 'tiny-model')
        self.assertTrue(calls)
        for command, env in calls:
            self.assertEqual(env['S1_HF_MODEL'], 'org/tiny-model')
            self.assertEqual(command[command.index('--adapter') + 1], str(cli.HF_TEMPLATE.resolve()))
            self.assertEqual(command[command.index('--model-label') + 1], 'tiny-model')

    def test_hf_model_problem_stops_eval_before_any_output(self):
        name = 'runs/.test-hf-unloadable'
        with patch.object(cli, 'hf_model_problem', return_value='gated repo'), \
                patch.object(cli.subprocess, 'run') as run:
            with self.assertRaises(SystemExit) as stop:
                cli.main(['eval', '--adapter', 'hf:org/gated-model', '--tasks', 'false_belief_story_qa',
                          '--output', name])
        self.assertEqual(str(stop.exception), 'hf:org/gated-model: gated repo')
        self.assertFalse((ROOT / name).exists())
        run.assert_not_called()

    def test_hf_model_problem_reads_the_config_only(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest('transformers is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'config.json').write_text(json.dumps({'model_type': 'gpt2'}))
            self.assertIsNone(cli.hf_model_problem(cli.adapter_spec('hf:' + tmp)))
            Path(tmp, 'config.json').write_text(json.dumps({'model_type': 'no_such_architecture'}))
            self.assertIn('no_such_architecture', cli.hf_model_problem(cli.adapter_spec('hf:' + tmp)))

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