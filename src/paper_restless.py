"""Original CogBench V1 RestlessBandit with separate action/confidence callbacks."""
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import traceback

import numpy as np
import pandas as pd
import torch

from src.paper_horizon import load_agent, write_json
from src.paper_twostep import _source_methods

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/CogBench/Experiments/RestlessBandit'
N_RUNS = 10
N_TRIALS = 80
SEED = 20261009


def load_official():
    spec = importlib.util.spec_from_file_location('original_restless_env', SOURCE / 'envs/bandit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class TracedEnv(module.MetacognitionEnv):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            self.instances.append(self)

        def step(self, action):
            before = list(map(float, self.mean_rewards))
            trial, block = self.t, self.block
            result = super().step(action)
            self.trace.append({'trial': trial, 'block_before': block, 'action': int(action),
                               'reward': float(result[1]), 'means_before': before,
                               'means_after': list(map(float, self.mean_rewards)),
                               'block_after': self.block,
                               'accuracy_before_step': int(action == int(np.argmax(before))),
                               'accuracy_after_step_original': int(action == int(np.argmax(self.mean_rewards)))})
            return result

    cls = _source_methods(SOURCE / 'query.py', 'MetacognitionExpForLLM',
                          ['run_single_experiment', 'set_experiment', 'keep_arms',
                           'del_letters_at_end', 'rdm_choice_agent', 'rdm_confidence_agent'])
    cls.set_experiment.__globals__['MetacognitionEnv'] = TracedEnv
    task = cls()
    task.arms = ['F', 'J']
    task.parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(version_number='1'))
    return task, TracedEnv


def official_score(frame):
    cls = _source_methods(SOURCE / 'store.py', 'StoringMetaCognitiveScores', ['get_scores'])
    columns = ['engine', 'run', 'performance_score1', 'performance_score1_name',
               'behaviour_score1', 'behaviour_score1_name', 'behaviour_score2',
               'behaviour_score2_name', 'performance_score2', 'performance_score2_name',
               'behaviour_score3', 'behaviour_score3_name']
    storing = pd.DataFrame(columns=columns)
    # The published StoringScores loop scores each original run separately;
    # in particular adjusted QSR must not share min/max across runs.
    for run, run_frame in frame.groupby('run', sort=True):
        storing = cls().get_scores(run_frame, storing, 'external_typed_choice_confidence', int(run))
    return storing


class ChoiceConfidenceLLM:
    Q_A = ('\n\nQ:', '\n\nA:')

    def __init__(self, agent, requests, answers):
        self.agent = agent
        self.requests, self.answers = requests, answers
        self.records = []

    def begin_run(self, run):
        self.run = run
        self.trial = 0
        self.chosen = None
        self.agent_rng = np.random.RandomState(SEED + 10000 + run)

    def generate(self, prompt):
        choosing = self.format_answer == 'Machine '
        phase = 'choice' if choosing else 'confidence'
        record_id = f'{self.run}/{self.trial}/{phase}'
        self.requests.write(json.dumps({'id': record_id, 'state': prompt,
                                       'original_specifications': self.default_query_specifications,
                                       'original_answer_prefix': self.format_answer}, allow_nan=False) + '\n')
        env_state = np.random.get_state()
        np.random.set_state(self.agent_rng.get_state())
        started = time.monotonic()
        try:
            # Rewards are drawn with CPU torch.normal in the original env.
            # Callback random draws cannot advance that environment stream.
            with torch.random.fork_rng(devices=[]):
                if choosing:
                    keys = ['F', 'J']
                    question = {'type': 'choice', 'instructions': prompt.rsplit('\n\n', 1)[-1],
                                'criteria': {key: f'Machine {key}' for key in keys}}
                    p = np.asarray(self.agent.choice_probabilities(prompt, question, keys), dtype=float)
                    if (p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any()
                            or (p > 1).any() or abs(p.sum() - 1) > 1e-8):
                        raise ValueError('Invalid action probabilities; no random fallback.')
                    native = getattr(self.agent, 'last_native_choice', None)
                    choice = native if native is not None else keys[int(p.argmax())]
                    if choice not in keys:
                        raise ValueError('Illegal native Restless choice.')
                    self.chosen = choice
                    row = {'id': record_id, 'run': self.run, 'trial': self.trial, 'phase': phase,
                           'choice': choice, 'probabilities': dict(zip(keys, p.tolist())),
                           'action_source': 'native_choice' if native is not None else 'probability_argmax'}
                    output = choice
                else:
                    if self.format_answer != f'Machine {self.chosen} with confidence 0.':
                        raise ValueError('Unexpected original confidence prefix.')
                    q = float(self.agent.confidence_probability(
                        prompt, self.chosen, self.default_query_specifications))
                    if not np.isfinite(q) or not 0 <= q <= 1:
                        raise ValueError('Invalid independent confidence report.')
                    # Original fixed answer prefix is 0. and asks for two decimals.
                    # Keep raw q separately; 1.00 cannot be represented by that
                    # prefix and is explicitly capped at .99 in this interface.
                    digits = min(99, int(np.round(q * 100)))
                    output = f'{digits:02d}'
                    row = {'id': record_id, 'run': self.run, 'trial': self.trial, 'phase': phase,
                           'choice': self.chosen, 'raw_reported_probability': q,
                           'original_two_decimal_report': digits / 100,
                           'endpoint_capped': int(np.round(q * 100)) > 99}
                    self.trial += 1
        finally:
            self.agent_rng.set_state(np.random.get_state())
            np.random.set_state(env_state)
        row['inference_seconds'] = time.monotonic() - started
        fields = ['last_input_tokens', 'last_prompt']
        if phase == 'confidence':
            fields.append('last_raw_answer')
        for name in fields:
            if hasattr(self.agent, name):
                row[name.removeprefix('last_')] = getattr(self.agent, name)
        self.records.append(row)
        self.answers.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        return output


def verify_run(frame, trace, records):
    if len(frame) != 80 or frame.trial.tolist() != list(range(80)) or len(records) != 160:
        raise ValueError('Original full 80-trial, two-response run required.')
    if frame.groupby('block').size().tolist() != [21, 20, 20, 19]:
        raise ValueError('Original actual block schedule changed.')
    if frame.choice.tolist() != [0 if r['choice'] == 'F' else 1 for r in records[::2]]:
        raise ValueError('Callback choices differ from executed actions.')
    if frame.accurate.tolist() != [t['accuracy_after_step_original'] for t in trace]:
        raise ValueError('Original post-step accuracy formula changed.')
    if not np.allclose(frame.confidence, [r['original_two_decimal_report'] for r in records[1::2]], atol=0, rtol=0):
        raise ValueError('Original source confidence parser differs from the recorded report.')
    if frame.reward.tolist() != [t['reward'] for t in trace]:
        raise ValueError('Executed rewards and original rows differ.')


def run_restless(output, adapter_file, device='cpu'):
    started = time.monotonic()
    config = {'status': 'loading', 'task': 'restless-generative', 'version': '1',
              'source': str(SOURCE), 'adapter_file': str(Path(adapter_file).resolve()), 'device': device,
              'data_kind': 'program_generated_original_environment_not_human_data',
              'num_runs': 10, 'trials_per_run': 80, 'expected_choice_queries': 800,
              'expected_confidence_queries': 800, 'seed': SEED,
              'history': 'Unchanged original complete current-run history of own actions, confidence and feedback.',
              'visible_information': 'Original prompt only; no hidden means, true block boundaries or future rewards.',
              'actual_environment': 'Original fixed blocks21/20/20/19; Gaussian rewards unbounded despite prompt range.',
              'seed_adaptation': 'New NumPy and CPU-Torch per-run environment seeds; callback streams isolated.',
              'confidence_adaptation': 'Separate chosen-current-best confidence/event query, never action p. '
                  'Original 0. prefix/two decimals retained; rounded1.00 capped.99 and raw value separately logged. '
                  'The original wording also anchors0 as guess, so confidence scores are not a universal probability calibration claim.',
              'accuracy_source_bug': 'Original query scores accuracy after env.step, so four block-final rows use next block means. '
                  'Preserve original scorer; separately report pre-step current-best accuracy/QSR.',
              'timing_context': 'Shared device; no exclusive-device performance claim.'}
    write_json(output / 'config.json', config)
    write_json(output / 'status.json', {'status': 'loading', 'expected_runs': N_RUNS})
    llm = None
    try:
        agent = load_agent(adapter_file, device)
        config.update(status='running', model_metadata=getattr(agent, 'metadata', {}),
                      load_seconds=time.monotonic() - started)
        write_json(output / 'config.json', config)
        task, env_class = load_official()
        frames = []
        with (output / 'requests.jsonl').open('w') as requests, (output / 'answers.jsonl').open('w') as answers, (output / 'targets.jsonl').open('w') as targets:
            llm = ChoiceConfidenceLLM(agent, requests, answers)
            for run in range(N_RUNS):
                llm.begin_run(run)
                np.random.seed(SEED + run)
                torch.manual_seed(SEED + run)
                before = len(llm.records)
                frame = task.run_single_experiment(llm)
                trace = env_class.instances[-1].trace
                verify_run(frame, trace, llm.records[before:])
                frame['run'] = run
                frame['accurate_before_step_additional'] = [t['accuracy_before_step'] for t in trace]
                frame.to_csv(output / f'run_{run}.csv', index=False)
                for target in trace:
                    targets.write(json.dumps({'run': run, **target}, allow_nan=False) + '\n')
                frames.append(frame)
                requests.flush(); answers.flush(); targets.flush()
                progress = {'status': 'running', 'completed_runs': run + 1, 'expected_runs': N_RUNS,
                            'completed_trials': len(frames) * N_TRIALS,
                            'completed_queries': len(llm.records), 'elapsed_seconds': time.monotonic() - started}
                write_json(output / 'status.json', progress)
                print(json.dumps(progress), flush=True)
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(output / 'trials.csv', index=False)
        scores = official_score(result)
        scores.to_csv(output / 'official_scores.csv', index=False)
        q = result.confidence.to_numpy()
        pre = result.accurate_before_step_additional.to_numpy()
        metric = scores[['performance_score1', 'performance_score2', 'behaviour_score1',
                         'behaviour_score2', 'behaviour_score3']].astype(float).mean()
        summary = {'status': 'complete', 'task': 'restless-generative', 'runs': len(frames),
                   'trials': len(result), 'queries': len(llm.records),
                   'mean_reward_original': float(metric.performance_score1),
                   'accuracy_post_step_original': float(metric.performance_score2),
                   'qsr_original': float(metric.behaviour_score1),
                   'mean_confidence_original': float(metric.behaviour_score2),
                   'adjusted_qsr_original': float(metric.behaviour_score3),
                   'aggregation': 'Original per-run scorer on each80-trial run; mean of ten run scores.',
                   'accuracy_pre_step_additional': float(pre.mean()),
                   'qsr_pre_step_additional': float((1 - (pre - q) ** 2).mean()),
                   'block_final_accuracy_mismatches': int((result.accurate != pre).sum()),
                   'confidence_endpoints_capped': sum(r.get('endpoint_capped', False) for r in llm.records),
                   'action_counts': result.choice.value_counts().to_dict(),
                   'maximum_input_tokens': getattr(agent, 'max_input_tokens', None),
                   'total_seconds': time.monotonic() - started, 'model_metadata': config['model_metadata'],
                   'interface_adaptation': config['confidence_adaptation'],
                   'timing_context': config['timing_context'],
                   'checks': 'All10x80 original trials and1600 separate action/confidence callbacks; '
                       'original rewards, legal actions, block schedule, confidence parser and post-step scorer preserved.'}
        if len(result) != 800 or len(llm.records) != 1600:
            raise ValueError('Full original Restless matrix is incomplete.')
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        failure = {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc(),
                   'completed_queries': len(llm.records) if llm else 0}
        write_json(output / 'status.json', failure)
        write_json(output / 'failure.json', failure)
        raise
