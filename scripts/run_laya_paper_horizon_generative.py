#!/usr/bin/env python3
"""Full Namazova Horizon rollout using official dynamics and Laya's choice API."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = ROOT / 'external/paper_matched_centaur_evaluation/horizon'
KEYS = ['participant_id', 'block', 'game', 'trial_num']


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_official():
    # Only bypass the obsolete login helper, as in run_paper_horizon.py.
    helper = types.ModuleType('get_models')
    previous = sys.modules.get('get_models')
    sys.modules['get_models'] = helper
    try:
        spec = importlib.util.spec_from_file_location(
            'published_laya_horizon', UPSTREAM / 'horizon_centaur_one.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['get_models']
        else:
            sys.modules['get_models'] = previous
    return module


class LayaPipeline:
    """The exact text-generation return interface consumed by the original loop."""

    def __init__(self, model, question, request_file):
        self.model = model
        self.question = question
        self.request_file = request_file
        self.records = []

    def __call__(self, prompt):
        import numpy as np
        started = time.monotonic()
        probability = self.model.choice_probabilities(prompt, self.question, ['H', 'I'])
        seconds = time.monotonic() - started
        # fix_seed in the original participant loop seeds NumPy. This is an
        # explicit legal-action sampling adapter, not LLM vocabulary sampling.
        choice = str(np.random.choice(['H', 'I'], p=probability))
        self.records.append({
            'query_index': len(self.records), 'choice': choice,
            'sdk_raw_probability_H': self.model.last_raw_probabilities[0],
            'sdk_raw_probability_I': self.model.last_raw_probabilities[1],
            'legal_probability_H': float(probability[0]),
            'legal_probability_I': float(probability[1]),
            'native_choice': self.model.last_native_choice,
            'input_tokens': self.model.last_input_tokens, 'inference_seconds': seconds,
        })
        self.request_file.write(json.dumps({
            'query_index': len(self.records) - 1, 'state': prompt,
            'question': self.question, 'sampled_action': choice,
        }, ensure_ascii=False) + '\n')
        return [{'generated_text': prompt + choice}]


def verify_rollout(timeline, result):
    """Detect omitted trials, reward misrouting, or human-history leakage."""
    import ast
    import numpy as np
    import pandas as pd
    expected = timeline.rename(columns={'trial_num_block': 'trial_num'}).set_index(KEYS).sort_index()
    actual = result.set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    if not actual.choice.isin(['H', 'I']).all():
        raise ValueError('Illegal action in full rollout.')
    free = actual.is_free.astype(bool)
    np.testing.assert_array_equal(free, expected.type.eq('free'))
    rewards = np.where(actual.choice.eq('H'), expected.reward_H, expected.reward_I)
    np.testing.assert_array_equal(actual.loc[free, 'reward'], rewards[free])
    np.testing.assert_array_equal(actual.loc[~free, 'choice'], expected.loc[~free, 'choice'])
    np.testing.assert_array_equal(actual.loc[~free, 'reward'], expected.loc[~free, 'reward'])
    for _, game in result.groupby(['participant_id', 'block', 'game'], sort=False):
        game = game.sort_values('trial_num')
        history = []
        cumulative = 0
        for row in game.itertuples():
            cumulative += row.reward
            if row.is_free:
                text_list = ast.literal_eval(row.prompt.removesuffix('You press <<'))
                observed_history = text_list[9:]
                if observed_history != history:
                    raise ValueError('Prompt is not the exact own-action/observed-reward history.')
                if row.cumulative_reward != cumulative:
                    raise ValueError('Incorrect free-trial cumulative reward.')
                history.append(f'You press <<{row.choice}>> and get {row.reward} points.')
            else:
                history.append(f'You are instructed to press {row.choice} and get {row.reward} points.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import pandas as pd
    import torch
    from audit_laya_paper_horizon_lengths import functions_from_source
    from src.choice_models import LayaChoice

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    timeline = pd.read_csv(UPSTREAM / 'data/in/timeline_structure.csv')
    counts = (timeline.participant_id.nunique(), len(timeline), int(timeline.type.eq('free').sum()))
    if counts != (31, 74400, 34720):
        raise ValueError(f'Expected the full original timeline, got {counts}.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Write results within this project.')
    output.mkdir(parents=True, exist_ok=False)
    question = functions_from_source(ROOT / 'src/horizon.py', ['choice_question'])['choice_question'](
        labels=('H', 'I'))
    official = load_official()
    seeds = official.generate_seeds(num_seeds=32)
    config = {
        'status': 'loading_model', 'model': 'Laya English', 'mode': 'generative',
        'protocol': 'original_namazova_timeline_and_loop_with_laya_typed_choice_sampling',
        'source_status': 'new_run_typed_choice_adapter',
        'paper': 'https://arxiv.org/abs/2508.07887',
        'source_csv': str(UPSTREAM / 'data/in/timeline_structure.csv'),
        'upstream_script': str(UPSTREAM / 'horizon_centaur_one.py'),
        'n_participants': 31, 'expected_free_choices': 34720,
        'question': question, 'output_key_order': ['H', 'I'],
        'action_rule': 'numpy.random.choice using normalized returned SDK probabilities',
        'participant_seeds': {str(int(p)): seeds[int(p) - 1] for p in timeline.participant_id.unique()},
        'history': 'Exact official per-game prompt with own previous actions and observed feedback.',
        'interface_adaptation': 'SDK typed-choice head and legal-action sampler replace text generation.',
        'sampling_difference': 'Laya legal categorical sampling has no vocabulary top_p/top_k; not an identical decoder.',
        'probability_processing': 'Preserve four-decimal SDK probabilities and normalize over H/I.',
        'precision': 'FP32 weights and forward, unchanged from predictive run',
        'device': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'timing_context': 'Shared device; timings are not exclusive-device performance.',
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pandas']},
    }
    save_json(output / 'config.json', config)
    started = time.monotonic()
    model = LayaChoice(device=args.device)
    model.agent.dtype = torch.float32
    config.update(status='running', load_seconds=time.monotonic() - started,
                  actual_parameter_dtypes=sorted({str(p.dtype) for p in model.agent.model.parameters()}),
                  actual_agent_autocast_dtype=str(model.agent.dtype), sdk_config=model.agent.cfg)
    save_json(output / 'config.json', config)
    print(json.dumps({'status': 'model_loaded', 'load_seconds': config['load_seconds']}), flush=True)
    frames = []
    scored_frames = []
    with (output / 'requests.jsonl').open('w') as requests:
        pipe = LayaPipeline(model, question, requests)
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
                raise ValueError('Laya queries do not align with official free-choice results.')
            scored = pd.concat([free[KEYS + ['reward']], queries], axis=1)
            scored.to_csv(output / f'probabilities_{int(participant)}.csv', index=False)
            frames.append(result)
            scored_frames.append(scored)
            requests.flush()
            progress = {
                'status': 'running', 'completed_participants': len(frames), 'total_participants': 31,
                'completed_free_choices': len(pipe.records), 'expected_free_choices': 34720,
                'elapsed_seconds': time.monotonic() - started,
                'max_input_tokens': model.max_input_tokens,
            }
            save_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    results = pd.concat(frames, ignore_index=True)
    free = results[results.is_free.astype(bool)].copy()
    if len(results) != 74400 or len(free) != 34720 or len(pipe.records) != 34720:
        raise ValueError('Incomplete full original timeline.')
    free['optimal'] = ((free.choice.eq('H') & free.reward_mean_H.gt(free.reward_mean_I)) |
                       (free.choice.eq('I') & free.reward_mean_I.gt(free.reward_mean_H)))
    by_trial = free.groupby(['participant_id', 'horizon', 'trial_num']).optimal.mean()
    by_trial.groupby(['horizon', 'trial_num']).agg(['mean', 'sem']).to_csv(output / 'optimal_choice_by_trial.csv')
    pd.concat(scored_frames, ignore_index=True).to_csv(output / 'trials.csv', index=False)
    summary = {
        'status': 'complete', 'model': 'Laya English', 'mode': 'generative',
        'source_status': 'new_run_typed_choice_adapter',
        'n_participants': 31, 'n_free_choices': len(free), 'n_rows': len(results),
        'mean_reward_free_additional_metric': float(free.reward.mean()),
        'mean_optimal_choice_rate': float(free.optimal.mean()),
        'max_input_tokens': model.max_input_tokens,
        'total_seconds': time.monotonic() - started, 'load_seconds': config['load_seconds'],
        'timing_context': config['timing_context'],
        'checks': 'Complete original keys; all rewards from sampled actions; every prompt has exact own history; no truncation.',
        'notes': 'Original environment loop and prompts; explicit typed-choice sampling adaptation. No human-prediction score.',
    }
    save_json(output / 'summary.json', summary)
    save_json(output / 'progress.json', summary)
    config['status'] = 'complete'
    save_json(output / 'config.json', config)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
