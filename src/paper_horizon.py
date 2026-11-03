"""External typed-choice agents in the complete original Namazova Horizon loop."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_KEYS = ['H', 'I']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_agent(adapter_file, device):
    """Load a trusted local Python file defining create_agent(device)."""
    spec = importlib.util.spec_from_file_location('benchmark_external_agent', adapter_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_agent(device)


class ChoicePipeline:
    """Supply the generation interface expected by the unchanged official loop."""

    def __init__(self, agent, question, request_file, probability_file, action_rule):
        self.agent = agent
        self.question = question
        self.request_file = request_file
        self.probability_file = probability_file
        self.action_rule = action_rule
        self.records = []

    def __call__(self, prompt):
        import numpy as np
        started = time.monotonic()
        probability = np.asarray(self.agent.choice_probabilities(
            prompt, self.question, OUTPUT_KEYS.copy()), dtype=float)
        seconds = time.monotonic() - started
        if probability.shape != (2,) or not np.isfinite(probability).all():
            raise ValueError('Return two finite probabilities in output_keys order.')
        if (probability < 0).any() or (probability > 1).any() or abs(probability.sum() - 1) > 1e-8:
            raise ValueError('Return normalized probabilities in [0,1]; the runner does not normalize.')
        native_choice = getattr(self.agent, 'last_native_choice', None)
        if native_choice is not None and native_choice not in OUTPUT_KEYS:
            raise ValueError('last_native_choice must be H or I when supplied.')
        if self.action_rule == 'sample':
            # The official participant loop seeds NumPy before each timeline.
            choice = str(np.random.choice(OUTPUT_KEYS, p=probability))
            source = 'external_probability_sampling'
        elif self.action_rule == 'native':
            if native_choice is not None:
                choice = native_choice
                source = 'native_choice'
            else:
                choice = OUTPUT_KEYS[int(probability.argmax())]
                source = 'submitted_probability_argmax'
        else:
            raise ValueError('action_rule must be native or sample.')
        query_index = len(self.records)
        record = {
            'query_index': query_index, 'choice': choice,
            'legal_probability_H': float(probability[0]),
            'legal_probability_I': float(probability[1]),
            'native_choice': native_choice, 'action_source': source,
            'inference_seconds': seconds,
        }
        if hasattr(self.agent, 'last_input_tokens'):
            record['input_tokens'] = int(self.agent.last_input_tokens)
        self.records.append(record)
        self.request_file.write(json.dumps({
            'id': str(query_index), 'state': prompt, 'question': self.question,
            'output_keys': OUTPUT_KEYS,
        }, ensure_ascii=False, allow_nan=False) + '\n')
        self.probability_file.write(json.dumps({
            'id': str(query_index), 'probabilities': dict(zip(OUTPUT_KEYS, probability.tolist())),
            'choice': choice, 'native_choice': native_choice, 'action_source': source,
            'inference_seconds': seconds,
        }, ensure_ascii=False, allow_nan=False) + '\n')
        return [{'generated_text': prompt + choice}]


def run_generative(output, adapter_file, device='cpu', action_rule='native'):
    """Run all 31 original timelines, with no sampled-data or shortened-run mode."""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Reuse the already verified official import and own-history/reward checks.
    from scripts.run_laya_paper_horizon_generative import KEYS, UPSTREAM, load_official, verify_rollout
    from scripts.audit_laya_paper_horizon_lengths import functions_from_source
    import pandas as pd

    timeline = pd.read_csv(UPSTREAM / 'data/in/timeline_structure.csv')
    counts = (timeline.participant_id.nunique(), len(timeline), int(timeline.type.eq('free').sum()))
    if counts != (31, 74400, 34720):
        raise ValueError(f'Expected the full original timeline, got {counts}.')
    question = functions_from_source(ROOT / 'src/horizon.py', ['choice_question'])['choice_question'](
        labels=('H', 'I'))
    official = load_official()
    seeds = official.generate_seeds(num_seeds=32)
    config = {
        'status': 'loading_agent', 'task': 'horizon-generative', 'mode': 'generative',
        'paper': 'https://arxiv.org/abs/2508.07887',
        'source_status': 'new_run_external_typed_choice_adapter',
        'source_csv': str(UPSTREAM / 'data/in/timeline_structure.csv'),
        'upstream_script': str(UPSTREAM / 'horizon_centaur_one.py'),
        'adapter_file': str(adapter_file.resolve()), 'device': device,
        'n_participants': 31, 'expected_rows': 74400, 'expected_free_choices': 34720,
        'question': question, 'output_key_order': OUTPUT_KEYS,
        'participant_seeds': {str(int(p)): seeds[int(p) - 1] for p in timeline.participant_id.unique()},
        'action_rule': action_rule,
        'native_action_definition': 'Use last_native_choice when non-null; otherwise argmax of submitted '
            'probabilities, with ties taking the first output_key. Each decision records its actual source.',
        'sample_action_definition': 'External numpy.random.choice of submitted probabilities, using original '
            'participant seeds; this is a separate execution condition from native.',
        'history': 'Exact original per-game prompt with own previous actions and observed feedback.',
        'interface_adaptation': 'state, typed question and ordered output_keys replace LLM text generation. '
            'Original environment and state prompt are reused; original decoder/inference is not reproduced.',
        'probability_processing': 'Record submitted probabilities unchanged; reject invalid values. No smoothing.',
        'input_processing': 'The runner forwards the complete original state unchanged. '
            'Encoding, precision and context handling inside the external adapter must be disclosed by its author.',
        'timing_context': 'Shared machine; measured elapsed time is not exclusive-device performance.',
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pandas']},
        'implementation_note': 'Add an external callback and reuse the full original simulate function; '
            'no change to existing model-specific runners, paper baselines or queued experiments.',
    }
    write_json(output / 'config.json', config)
    started = time.monotonic()
    agent = load_agent(adapter_file, device)
    config.update(status='running', load_seconds=time.monotonic() - started,
                  agent_type=f'{type(agent).__module__}.{type(agent).__qualname__}')
    write_json(output / 'config.json', config)
    print(json.dumps({'status': 'agent_loaded', 'load_seconds': config['load_seconds']}), flush=True)
    frames = []
    scored_frames = []
    with (output / 'requests.jsonl').open('w') as requests, (output / 'probabilities.jsonl').open('w') as probabilities:
        pipe = ChoicePipeline(agent, question, requests, probabilities, action_rule)
        for participant in timeline.participant_id.unique():
            begin = len(pipe.records)
            participant_data = timeline[timeline.participant_id == participant]
            result = official.simulate_participant_by_block(
                participant_data, participant, seeds, None, None, pipe)
            result.to_csv(output / f'participant_{int(participant)}.csv', index=False)
            verify_rollout(participant_data, result)
            free = result[result.is_free.astype(bool)].reset_index(drop=True)
            queries = pd.DataFrame(pipe.records[begin:])
            if len(queries) != len(free) or queries.choice.tolist() != free.choice.tolist():
                raise ValueError('External callback queries do not align with original free-choice results.')
            scored = pd.concat([free[KEYS + ['reward']], queries], axis=1)
            scored.to_csv(output / f'probabilities_{int(participant)}.csv', index=False)
            frames.append(result)
            scored_frames.append(scored)
            requests.flush()
            probabilities.flush()
            progress = {
                'status': 'running', 'completed_participants': len(frames), 'total_participants': 31,
                'completed_free_choices': len(pipe.records), 'expected_free_choices': 34720,
                'elapsed_seconds': time.monotonic() - started,
            }
            write_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    results = pd.concat(frames, ignore_index=True)
    free = results[results.is_free.astype(bool)].copy()
    if len(results) != 74400 or len(free) != 34720 or len(pipe.records) != 34720:
        raise ValueError('Incomplete full original timeline.')
    # Exact original notebook definition: choose the arm with strictly higher mean.
    free['optimal'] = ((free.choice.eq('H') & free.reward_mean_H.gt(free.reward_mean_I)) |
                       (free.choice.eq('I') & free.reward_mean_I.gt(free.reward_mean_H)))
    by_trial = free.groupby(['participant_id', 'horizon', 'trial_num']).optimal.mean()
    by_trial.groupby(['horizon', 'trial_num']).agg(['mean', 'sem']).to_csv(output / 'optimal_choice_by_trial.csv')
    trials = pd.concat(scored_frames, ignore_index=True)
    trials.to_csv(output / 'trials.csv', index=False)
    summary = {
        'status': 'complete', 'task': 'horizon-generative', 'mode': 'generative',
        'source_status': config['source_status'], 'adapter_file': config['adapter_file'],
        'action_rule': action_rule, 'action_source_counts': trials.action_source.value_counts().to_dict(),
        'n_participants': 31, 'n_free_choices': len(free), 'n_rows': len(results),
        'mean_reward_free_additional_metric': float(free.reward.mean()),
        'mean_optimal_choice_rate': float(free.optimal.mean()),
        'total_seconds': time.monotonic() - started, 'load_seconds': config['load_seconds'],
        'timing_context': config['timing_context'],
        'checks': 'Complete original keys; all rewards use selected actions; each prompt has exact own history; '
            'query count and action sequence match original-loop results.',
        'notes': 'Complete original environment with a new external typed-choice adapter. '
            'No human next-choice prediction/NLL or paper-baseline identity inferred.',
    }
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', summary)
    config['status'] = 'complete'
    write_json(output / 'config.json', config)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return summary
