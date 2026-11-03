"""Complete CogBench v1 two-step environment with an external choice callback.

The published task methods and scoring methods execute directly from their
local source. Typed probabilities and reproducible per-run seeds are explicitly
new interface/replay protocols, rather than the original LLM decoder.
"""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.paper_horizon import load_agent, write_json

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external/CogBench/Experiments/TwoStepTask'
N_RUNS = 100
N_TRIALS = 20
DATA_COLUMNS = ['trial', 'action1', 'state', 'action2', 'reward',
                'probsA', 'probsB', 'probsC', 'probsD', 'run']
SCORE_COLUMNS = ['engine', 'run', 'performance_score1', 'performance_score1_name',
                 'performance_score1_CI', 'behaviour_score1',
                 'behaviour_score1_name', 'behaviour_score1_CI']


def _source_methods(path, class_name, method_names):
    """Execute only the actual published methods, without unrelated API imports."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in method_names]
    if {node.name for node in selected} != set(method_names):
        raise ValueError(f'Published methods missing in {path}.')
    namespace = {'np': np, 'pd': pd, 'smf': smf}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return type(class_name, (), {name: namespace[name] for name in method_names})


def load_official():
    cls = _source_methods(
        UPSTREAM / 'query.py', 'TwoStepTaskExpForLLM',
        ['run_single_experiment', 'transition', 'random_fct', 'keep_arms'])
    task = cls()
    task.num_trials = N_TRIALS
    task.arms = ['X', 'Y']
    # Avoid changing process CLI arguments or invoking the original API loader.
    task.parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(version_number='1'))
    return task


def load_official_scorer():
    cls = _source_methods(
        UPSTREAM / 'store.py', 'StoringModelBasednessScores',
        ['get_scores', 'get_interaction_effect'])
    return cls()


def choice_question(prompt, keys):
    """The original current question, with names for the legal typed options."""
    instruction = prompt.rsplit('\n\nQ:', 1)[1].strip()
    noun = 'planet' if keys == ['X', 'Y'] else 'alien'
    return {'type': 'choice', 'instructions': instruction,
            'criteria': {key: f'{noun} {key}' for key in keys}}


class ChoiceLLM:
    """Implement the generation callback required by the unchanged task loop."""

    Q_A = ('\n\nQ:', '\n\nA:')

    def __init__(self, agent, requests, probabilities, action_rule='native'):
        if action_rule not in {'native', 'sample'}:
            raise ValueError('action_rule must be native or sample.')
        self.agent = agent
        self.requests = requests
        self.probabilities = probabilities
        self.action_rule = action_rule
        self.records = []

    def begin_run(self, run, action_seed):
        self.run = run
        self.run_query = 0
        self.action_rng = np.random.RandomState(action_seed)
        self.agent_rng = np.random.RandomState(action_seed + N_RUNS)

    def generate(self, prompt):
        trial, stage_index = divmod(self.run_query, 2)
        stage = stage_index + 1
        if stage == 1:
            if self.format_answer != 'Planet ':
                raise ValueError('Unexpected original first-stage answer prefix.')
            keys = ['X', 'Y']
        elif self.format_answer == 'Alien ':
            keys = ['D', 'F'] if prompt.endswith('alien D or F?\n') else ['J', 'K']
            if not prompt.endswith(f'alien {keys[0]} or {keys[1]}?\n'):
                raise ValueError('Unexpected original second-stage legal question.')
        else:
            raise ValueError('Unexpected original second-stage answer prefix.')
        question = choice_question(prompt, keys)
        record_id = f'{self.run}/{trial}/{stage}'
        self.requests.write(json.dumps({
            'id': record_id, 'run': self.run, 'trial': trial, 'stage': stage,
            'state': prompt, 'question': question, 'output_keys': keys,
            'original_llm_default_prompt': prompt + self.Q_A[1] + ' ' + self.format_answer,
        }, ensure_ascii=False, allow_nan=False) + '\n')
        # Agent NumPy randomness cannot advance the environment's RNG stream.
        environment_rng_state = np.random.get_state()
        np.random.set_state(self.agent_rng.get_state())
        started = time.monotonic()
        try:
            probability = np.asarray(self.agent.choice_probabilities(
                prompt, question, keys.copy()), dtype=float)
        finally:
            self.agent_rng.set_state(np.random.get_state())
            np.random.set_state(environment_rng_state)
        seconds = time.monotonic() - started
        if probability.shape != (2,) or not np.isfinite(probability).all():
            raise ValueError('Return two finite probabilities in output_keys order.')
        if ((probability < 0).any() or (probability > 1).any()
                or abs(probability.sum() - 1) > 1e-8):
            raise ValueError('Return normalized probabilities in [0,1]; no normalization is applied.')
        native_choice = getattr(self.agent, 'last_native_choice', None)
        if native_choice is not None and native_choice not in keys:
            raise ValueError('last_native_choice must be one of the current legal actions.')
        if self.action_rule == 'sample':
            choice = str(self.action_rng.choice(keys, p=probability))
            source = 'external_probability_sampling'
        elif native_choice is not None:
            choice, source = native_choice, 'native_choice'
        else:
            choice, source = keys[int(probability.argmax())], 'submitted_probability_argmax'
        record = {
            'id': record_id, 'run': self.run, 'trial': trial, 'stage': stage,
            'choice': choice, 'native_choice': native_choice, 'action_source': source,
            'probabilities': dict(zip(keys, probability.tolist())),
            'inference_seconds': seconds,
        }
        if hasattr(self.agent, 'last_input_tokens'):
            record['input_tokens'] = int(self.agent.last_input_tokens)
        if hasattr(self.agent, 'last_raw_required_tokens'):
            record['raw_required_tokens'] = int(self.agent.last_raw_required_tokens)
        if hasattr(self.agent, 'last_prompt'):
            record['actual_adapter_prompt'] = self.agent.last_prompt
        if hasattr(self.agent, 'last_label_mapping'):
            record['label_mapping'] = self.agent.last_label_mapping
        if hasattr(self.agent, 'last_allowed_token_mass'):
            record['allowed_token_mass'] = float(self.agent.last_allowed_token_mass)
        if hasattr(self.agent, 'last_raw_probabilities'):
            record['raw_adapter_probabilities_before_adapter_normalization'] = self.agent.last_raw_probabilities
        self.records.append(record)
        self.probabilities.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.run_query += 1
        return choice


def verify_run(frame, decisions):
    """Check complete trial/action alignment and original reward/rate domains."""
    if len(frame) != N_TRIALS or frame.trial.tolist() != list(range(N_TRIALS)):
        raise ValueError('Incomplete original 20-trial run.')
    if len(decisions) != 2 * N_TRIALS:
        raise ValueError('Each original trial requires two decisions.')
    if not frame.action1.isin(['X', 'Y']).all() or not frame.state.isin(['X', 'Y']).all():
        raise ValueError('Invalid first-stage action or observed state.')
    legal_second = ((frame.state.eq('X') & frame.action2.isin(['D', 'F'])) |
                    (frame.state.eq('Y') & frame.action2.isin(['J', 'K'])))
    if not legal_second.all() or not frame.reward.isin([0, 1]).all():
        raise ValueError('Invalid second-stage action or binary reward.')
    rates = frame[['probsA', 'probsB', 'probsC', 'probsD']].to_numpy()
    if not np.isfinite(rates).all() or (rates < 0.25).any() or (rates > 0.75).any():
        raise ValueError('Original drifting reward rates must remain in [0.25,0.75].')
    if ([row['choice'] for row in decisions[::2]] != frame.action1.tolist() or
            [row['choice'] for row in decisions[1::2]] != frame.action2.tolist()):
        raise ValueError('Callback actions do not match executed environment actions.')


def stay_data(frame):
    """Expose original per-run lagged rows; no transitions across run boundaries."""
    rows = []
    for run in frame.run.unique():
        data = frame[frame.run == run]
        actions = data.action1.to_numpy()
        states = data.state.to_numpy()
        for trial in range(len(data) - 1):
            reward = int(data.reward.iloc[trial])
            common = bool(actions[trial] == states[trial])
            rows.append({'run': int(run), 'trial': int(data.trial.iloc[trial]),
                         'reward': reward, 'common': common,
                         'stay': float(actions[trial + 1] == actions[trial]),
                         'reward_coded': 1 if reward else -1,
                         'common_coded': 1 if common else -1,
                         'interaction_coded': (1 if reward else -1) * (1 if common else -1)})
    return pd.DataFrame(rows)


def score(frame):
    """Execute original CogBench get_scores, including its OLS and trial CI."""
    table = load_official_scorer().get_scores(
        frame, pd.DataFrame(columns=SCORE_COLUMNS), 'external_typed_choice', run=0)
    return table, stay_data(frame)


def run_twostep(output, adapter_file, device='cpu', action_rule='native', seed=20261006):
    """Run fixed original version 1: 100 generated runs of 20 trials each."""
    output, adapter_file = Path(output), Path(adapter_file)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    config = {
        'status': 'loading_agent', 'task': 'twostep-generative', 'mode': 'generative',
        'data_kind': 'program_generated_environment_not_human_data',
        'paper': 'https://arxiv.org/abs/2402.18225',
        'source_status': 'original_CogBench_v1_environment_with_new_typed_choice_adapter',
        'upstream_query': str(UPSTREAM / 'query.py'), 'upstream_scoring': str(UPSTREAM / 'store.py'),
        'adapter_file': str(adapter_file.resolve()), 'device': device, 'action_rule': action_rule,
        'n_runs': N_RUNS, 'trials_per_run': N_TRIALS, 'expected_trials': N_RUNS * N_TRIALS,
        'expected_decisions': 2 * N_RUNS * N_TRIALS, 'version_number': '1',
        'transition_probabilities': {'chosen_planet': 0.7, 'other_planet': 0.3},
        'reward_rates': 'Original U(0.25,0.75) initialization; after each trial add N(0,0.025) '
                        'independently to all four rates and clip to [0.25,0.75].',
        'seed': seed, 'environment_seeds': {str(run): seed + run for run in range(N_RUNS)},
        'action_seeds': {str(run): seed + 10000 + run for run in range(N_RUNS)},
        'seeding_status': 'New explicit replay protocol; original CogBench script does not specify seeds. '
                          'Separate action and agent NumPy streams preserve environment RNG against callback use.',
        'history': 'Complete unchanged original per-run history of own choices, reached planets and rewards; '
                   'reset in original task loop after 20 trials. No hidden reward rates or future feedback in inputs.',
        'interface_adaptation': 'The unchanged original task prompt is state; its current question and legal '
                                'planet/alien options form a typed choice question. Original LLM decoder, '
                                'answer prefix, CoT and step-back inference are not reproduced.',
        'probability_processing': 'Record submitted legal-action probabilities unchanged; reject invalid '
                                  'values; they are not reward/event probabilities.',
        'native_action_definition': 'Supplied legal last_native_choice; otherwise submitted probability '
                                    'argmax, with ties taking first key. Record actual source each time.',
        'sample_action_definition': 'External categorical sampling on a separate per-run RandomState stream.',
        'input_processing': 'Runner never shortens history. Adapter encoding/context limits and precision '
                            'must be disclosed; fixed model configuration is not changed by the runner.',
        'official_metrics': 'Reward mean and 1.96 * trial sample SD / sqrt(n); original pooled OLS stay ~ '
                            'reward + common + interaction with reward/common coded -1/+1 and 1.96*OLS SE.',
        'timing_context': 'Shared machine; elapsed time is not exclusive-device performance.',
        'versions': {name: importlib.metadata.version(name)
                     for name in ['numpy', 'pandas', 'statsmodels']},
        'implementation_note': 'Execute original source methods, adding only an external callback, '
                               'full-input/action logs and explicit replay seeds. Existing runners unchanged.',
    }
    write_json(output / 'config.json', config)
    frames = []
    llm = None
    try:
        load_started = time.monotonic()
        agent = load_agent(adapter_file, device)
        config.update(status='running', load_seconds=time.monotonic() - load_started,
                      agent_type=f'{type(agent).__module__}.{type(agent).__qualname__}')
        if hasattr(agent, 'get_agent_metadata'):
            config['agent_metadata'] = agent.get_agent_metadata()
        elif hasattr(agent, 'metadata'):
            config['agent_metadata'] = agent.metadata
        write_json(output / 'config.json', config)
        print(json.dumps({'status': 'agent_loaded', 'load_seconds': config['load_seconds']}), flush=True)
        official = load_official()
        with (output / 'requests.jsonl').open('w') as requests, (output / 'probabilities.jsonl').open('w') as probabilities:
            llm = ChoiceLLM(agent, requests, probabilities, action_rule)
            for run in range(N_RUNS):
                begin = len(llm.records)
                llm.begin_run(run, seed + 10000 + run)
                np.random.seed(seed + run)
                result = official.run_single_experiment(llm)
                result['run'] = run
                result = result[DATA_COLUMNS]
                verify_run(result, llm.records[begin:])
                result.to_csv(output / f'run_{run}.csv', index=False)
                frames.append(result)
                requests.flush()
                probabilities.flush()
                progress = {'status': 'running', 'completed_runs': len(frames), 'total_runs': N_RUNS,
                            'completed_trials': len(frames) * N_TRIALS,
                            'completed_decisions': len(llm.records), 'expected_decisions': 4000,
                            'elapsed_seconds': time.monotonic() - started}
                write_json(output / 'progress.json', progress)
                if run == 0 or (run + 1) % 10 == 0:
                    print(json.dumps(progress), flush=True)
        result = pd.concat(frames, ignore_index=True)
        if len(result) != 2000 or len(llm.records) != 4000:
            raise ValueError('Incomplete original 100 by 20 protocol.')
        result.to_csv(output / 'trials.csv', index=False)
        by_run = result.groupby('run').reward.agg(['sum', 'mean', 'count']).reset_index()
        by_run.to_csv(output / 'run_rewards.csv', index=False)
        scores, stay = score(result)
        scores.to_csv(output / 'official_scores.csv', index=False)
        stay.to_csv(output / 'stay_regression_data.csv', index=False)
        trials = pd.DataFrame(llm.records)
        trials.to_csv(output / 'decisions.csv', index=False)
        metric = scores.iloc[0]
        summary = {
            'status': 'complete', 'task': config['task'], 'mode': 'generative',
            'source_status': config['source_status'], 'data_kind': config['data_kind'],
            'adapter_file': config['adapter_file'], 'action_rule': action_rule, 'seed': seed,
            'n_runs': len(frames), 'n_trials': len(result), 'n_decisions': len(llm.records),
            'total_rewards': int(result.reward.sum()),
            'mean_reward': float(metric.performance_score1),
            'mean_reward_ci_halfwidth_original_trial_formula': float(metric.performance_score1_CI),
            'stay_reward_common_interaction_original_OLS': float(metric.behaviour_score1),
            'stay_interaction_ci_halfwidth_original_OLS': float(metric.behaviour_score1_CI),
            'n_stay_observations': len(stay),
            'reward_mean_sem_across_generated_runs_additional': float(by_run['mean'].sem()),
            'action_source_counts': trials.action_source.value_counts().to_dict(),
            'first_stage_action_counts': result.action1.value_counts().to_dict(),
            'second_stage_action_counts': result.action2.value_counts().to_dict(),
            'observed_common_transition_rate': float(result.action1.eq(result.state).mean()),
            'total_inference_seconds': float(trials.inference_seconds.sum()),
            'total_seconds': time.monotonic() - started, 'load_seconds': config['load_seconds'],
            'timing_context': config['timing_context'],
            'checks': 'All 100 original 20-trial runs; 4,000 legal callback actions match environment '
                      'results; original binary rewards, rate bounds and source stay scoring preserved.',
            'notes': 'Program-generated runs are not human participants. No human behavioral NLL or '
                     'probability calibration claim. Original reward/stay metrics with new typed interface '
                     'and explicit replay seeds; original paper model decoder/settings remain separate.',
        }
        if 'input_tokens' in trials:
            summary['min_input_tokens'] = int(trials.input_tokens.min())
            summary['max_input_tokens'] = int(trials.input_tokens.max())
        write_json(output / 'summary.json', summary)
        write_json(output / 'progress.json', summary)
        config['status'] = 'complete'
        write_json(output / 'config.json', config)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    except Exception as error:
        failure = {'status': 'failed', 'task': config['task'], 'completed_runs': len(frames),
                   'completed_decisions': len(llm.records) if llm is not None else 0,
                   'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'total_seconds': time.monotonic() - started}
        config.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'config.json', config)
        write_json(output / 'failure.json', failure)
        write_json(output / 'progress.json', failure)
        raise
