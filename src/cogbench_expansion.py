"""Published CogBench BART, temporal discounting and instrumental learning.

Task methods execute from the archived upstream source. Typed option scoring,
explicit seeds and request logs are our interface protocol. No upstream files
are changed. Instrumental learning removes the two identified duplicate calls
and rows; its 96 real trials and original feedback remain intact.
"""
import ast
import importlib.util
import json
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/CogBench/Experiments'
TASK_DIR = {'temporal': 'TemporalDiscounting', 'bart': 'BART',
            'instrumental': 'InstrumentalLearning'}
DEFAULT_RUNS = {'temporal': 1, 'bart': 10, 'instrumental': 10}


def source_methods(path, class_name, names, extra=None):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != set(names):
        raise ValueError(f'Missing published methods in {path}')
    ns = {'np': np, 'pd': pd}
    ns.update(extra or {})
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    return type(class_name, (), {name: ns[name] for name in names})


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task(task_name, version='1', base_pay=500):
    path = SOURCE / TASK_DIR[task_name]
    if task_name == 'temporal':
        cls = source_methods(path / 'query.py', 'TemporalDiscuountingExpForLLM',
                             ['run_single_experiment', 'temporal_discounting',
                              'random_fct', 'del_letters_at_end'])
    elif task_name == 'bart':
        env = source_module(path / 'envs/bart_env.py', 'expansion_bart_env').Bart_env
        cls = source_methods(path / 'query.py', 'RiskExpForLLM',
                             ['run_single_experiment', 'random_fct', 'del_letters_at_end'],
                             {'Bart_env': env})
    else:
        from torch.distributions import Binomial
        cls = source_methods(path / 'query.py', 'TwoArmedBandit4Context_ExpForLLM',
                             ['reset', 'step'], {'Binomial': Binomial})
    task = cls()
    # The upstream BART constructor shuffles its list in place. Give each run
    # its own fresh original list instead of leaking permutations across runs.
    task.parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(
        version_number=version, base_pay=base_pay, trials=15, max_pumps=[128, 32, 8]))
    return task


def typed_question(task_name, prompt):
    instruction = prompt.rsplit('\n\nQ:', 1)[-1].strip()
    if task_name == 'temporal':
        pairs = re.findall(r'- Option ([12]): ([^\n]+)', instruction)
    elif task_name == 'bart':
        pairs = re.findall(r"Option ([12]) for '([^']+)'", instruction)
    else:
        letters = re.findall(r'Machine ([A-Z])', instruction)
        pairs = [(letter, f'Machine {letter}') for letter in letters]
    if len(pairs) != 2 or len({key for key, _ in pairs}) != 2:
        raise ValueError(f'Cannot read two original legal options: {instruction}')
    return {'type': 'choice', 'instructions': instruction, 'criteria': dict(pairs)}


class ChoiceLLM:
    Q_A = ('\n\nQ:', '\n\nA:')

    def __init__(self, agent, task_name, stream, action_rule='native', device='cpu'):
        self.agent, self.task_name, self.stream = agent, task_name, stream
        self.action_rule, self.device = action_rule, device
        self.records = []
        self.failed = False

    def begin_run(self, run, seed):
        self.run, self.query_index = run, 0
        self.action_rng = np.random.RandomState(seed)

    def generate(self, prompt):
        question = typed_question(self.task_name, prompt)
        keys = list(question['criteria'])
        record = {'run': self.run, 'query_index': self.query_index,
                  'state': prompt, 'question': question, 'output_keys': keys}
        import torch
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            p = np.asarray(self.agent.choice_probabilities(prompt, question, keys), dtype=float)
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
            if (p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any()
                    or (p > 1).any() or abs(p.sum() - 1) > 1e-8):
                raise ValueError('Expected two valid normalized option probabilities.')
            if self.action_rule == 'sample':
                choice = str(self.action_rng.choice(keys, p=p))
                source = 'external_probability_sampling'
            else:
                choice = getattr(self.agent, 'last_native_choice', keys[int(p.argmax())])
                source = 'native_choice_or_legal_probability_argmax'
            if choice not in keys:
                raise ValueError(f'Invalid native choice {choice!r}')
            record.update(choice=choice, probabilities=dict(zip(keys, p.tolist())),
                          action_source=source, valid=True)
            for attr in ['last_input_tokens', 'last_raw_probabilities', 'last_prompt',
                         'last_label_mapping', 'last_allowed_token_mass']:
                if hasattr(self.agent, attr):
                    record[attr.removeprefix('last_')] = getattr(self.agent, attr)
        except Exception as exc:
            self.failed = True
            record.update(valid=False, error=repr(exc))
            raise
        finally:
            record['inference_seconds_shared_device'] = time.perf_counter() - started
            self.records.append(record)
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.stream.flush()
            self.query_index += 1
        return choice


def run_instrumental(task, bridge):
    """Original 96-trial query with duplicate generation/writes removed."""
    env_cls = source_module(SOURCE / 'InstrumentalLearning/envs/bandits.py',
                            'expansion_instrumental_env').OptimismBiasTaskPalminteri
    env = env_cls(num_actions=2, reward_scaling=1, max_steps_per_context=24, num_contexts=4)
    env.reset()
    task.Q_, task.A_ = bridge.Q_A
    instructions, history, trials_left, question = task.reset(
        env, env.contexts[0, 0], env.max_steps)
    current = env.contexts[0, 0]
    rows = []
    done = False
    for t in range(env.max_steps):
        prompt = instructions + trials_left + '\n' + history + '\n' + question
        action = bridge.generate(prompt)
        action_int = int(env.arms[int(current) - 1][action])
        reward = env.rewards[0, t, action_int].item()
        rows.append([t, int(current), env.mean_rewards[0, t, 0].item(),
                     env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),
                     env.rewards[0, t, 1].item(), action_int, reward])
        if not done:
            history, trials_left, current, question, done = task.step(
                env, history, current, action, t)
    return pd.DataFrame(rows, columns=['trial', 'task', 'mean0', 'mean1',
                                       'reward0', 'reward1', 'choice', 'reward'])


def run_one(task_name, bridge, run, seed, version='1', base_pay=500):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    bridge.begin_run(run, seed + 1000000)
    task = load_task(task_name, version, base_pay)
    df = (run_instrumental(task, bridge) if task_name == 'instrumental'
          else task.run_single_experiment(bridge))
    if bridge.failed or df.isna().any().any():
        raise RuntimeError('Official task failed; request failure preserved; no random replacement.')
    expected = 1 if task_name == 'temporal' else (96 if task_name == 'instrumental'
                                                 else (45 if version == '1' else 90))
    if len(df) != expected:
        raise ValueError(f'Expected {expected} official rows; received {len(df)}.')
    df['run'] = run
    return df


def score_runs(task_name, df, model_label, fit_seed=0):
    """Execute original scores. Instrumental DE bounds/maxiter are unchanged."""
    from scipy.optimize import differential_evolution
    import torch
    import torch.nn.functional as F
    path = SOURCE / TASK_DIR[task_name] / 'store.py'
    names = {'bart': 'StoringRiskScores', 'temporal': 'StoringTemporalDiscountingScores',
             'instrumental': 'StoringOptimismBias_and_LR'}
    methods = ['get_scores']
    if task_name == 'instrumental':
        methods += ['rescorla_wagner', 'nll', 'fit_parameters']
    cls = source_methods(path, names[task_name], methods,
                         {'torch': torch, 'F': F, 'differential_evolution': differential_evolution})
    scorer = cls()
    n_behaviour = 2 if task_name == 'bart' else 4
    columns = ['engine', 'run', 'performance_score1', 'performance_score1_name']
    for i in range(1, n_behaviour + 1):
        columns.extend([f'behaviour_score{i}', f'behaviour_score{i}_name'])
    scorer.parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(columns=columns[6:]))
    table = pd.DataFrame(columns=columns)
    for run, rows in df.groupby('run', sort=True):
        np.random.seed(fit_seed + int(run))
        table = scorer.get_scores(rows.reset_index(drop=True), table, model_label, int(run))
    return table


def summary_metrics(task_name, df, records):
    good = [r for r in records if r['valid']]
    result = {'completed_runs': int(df.run.nunique()), 'data_rows': len(df),
              'model_calls': len(records), 'invalid_response_rate':
              (len(records) - len(good)) / len(records) if records else None}
    if good:
        latency = [r['inference_seconds_shared_device'] for r in good]
        result.update(latency_p50_seconds=float(np.percentile(latency, 50)),
                      latency_p95_seconds=float(np.percentile(latency, 95)))
    if task_name in {'bart', 'instrumental'}:
        result['mean_reward'] = float(df.reward.mean())
    if task_name == 'instrumental':
        chosen = np.where(df.choice.to_numpy() == 0, df.mean0, df.mean1)
        result['pseudo_regret_total'] = float((np.maximum(df.mean0, df.mean1) - chosen).sum())
    return result
