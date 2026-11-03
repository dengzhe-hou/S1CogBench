#!/usr/bin/env python3
"""Full published Horizon test with intact prompts and a Laya typed-choice head."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-reference', type=Path,
                        help='Compare 16 already completed CPU requests before the complete new run.')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import numpy as np
    import pandas as pd
    import torch
    from audit_laya_paper_horizon_lengths import functions_from_source
    from src.choice_models import LayaChoice

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    official = ROOT / 'external/paper_matched_centaur_evaluation/horizon'
    helpers = functions_from_source(official / 'predictive_horizon_centaur.py',
                                    ['build_game_intro', 'format_forced_trials', 'format_past_trials'])
    question = functions_from_source(ROOT / 'src/horizon.py', ['choice_question'])['choice_question'](
        labels=('H', 'I'))
    data = pd.read_csv(official / 'data/in/test_data.csv')
    if set(data.participant_id) != {1, 2, 7, 23, 26, 31} or int(data.type.eq('free').sum()) != 6720:
        raise ValueError('Expected the complete published six-participant test.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Write results within this project.')
    output.mkdir(parents=True, exist_ok=False)
    config = {
        'status': 'loading_model', 'model': 'local_laya_english',
        'protocol': 'namazova_original_test_and_centaur_prompts_with_laya_typed_choice_adapter',
        'paper': 'https://arxiv.org/abs/2508.07887',
        'source_csv': str(official / 'data/in/test_data.csv'),
        'n_participants': 6, 'expected_free_choices': 6720,
        'device': args.device, 'cpu_threads': 4, 'inference': 'SDK calibrated choice probabilities; no action rollout',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'precision': 'FP32 weights and forward; preserved from CPU execution',
        'timing_context': ('shared with this project Centaur70B evaluation; not exclusive GPU latency'
                           if args.device.startswith('cuda') else 'CPU four threads'),
        'question': question, 'output_key_order': ['H', 'I'],
        'history': 'Unmodified official per-game prompt; previous human choices and observed rewards only.',
        'metric': 'legal_conditioned_nll_eps1e8', 'epsilon': 1e-8,
        'original_full_vocabulary_nll': None,
        'original_full_vocabulary_nll_status': 'not_defined_for_typed_choice_API',
        'probability_processing': 'Preserve rounded SDK probabilities; normalize their sum over H/I for scoring.',
        'interface_adaptation': 'SDK choice instruction/options/special tokens surround the complete original prompt.',
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'pandas', 'numpy']},
    }
    save_json(output / 'config.json', config)
    started = time.monotonic()
    model = LayaChoice(device=args.device)
    # The SDK normally switches CUDA autocast to bf16. Keep the CPU run's
    # FP32 arithmetic when changing only the execution device.
    model.agent.dtype = torch.float32
    load_seconds = time.monotonic() - started
    config.update({'status': 'running', 'load_seconds': load_seconds,
                   'sdk_config': model.agent.cfg,
                   'actual_parameter_dtypes': sorted({str(p.dtype) for p in model.agent.model.parameters()}),
                   'actual_agent_autocast_dtype': str(model.agent.dtype),
                   'actual_device': str(model.agent.device)})
    save_json(output / 'config.json', config)
    if args.cpu_reference:
        reference = args.cpu_reference.resolve()
        reference_rows = pd.read_csv(reference / 'trials.csv').head(16)
        with (reference / 'requests.jsonl').open() as handle:
            requests = [json.loads(next(handle)) for _ in range(len(reference_rows))]
        comparisons = []
        for request, (_, previous) in zip(requests, reference_rows.iterrows()):
            keys = ['participant_id', 'block', 'game', 'trial_num']
            if any(int(request[k]) != int(previous[k]) for k in keys):
                raise ValueError('CPU reference requests and probabilities are not aligned.')
            probability = model.choice_probabilities(request['state'], request['question'], ['H', 'I'])
            cpu_probability = np.array([previous.legal_probability_H, previous.legal_probability_I])
            comparisons.append({**{k: int(request[k]) for k in keys},
                                'cpu_probabilities': cpu_probability.tolist(),
                                'gpu_probabilities': probability.tolist(),
                                'max_absolute_probability_difference': float(abs(probability - cpu_probability).max()),
                                'native_choice_equal': model.last_native_choice == previous.native_choice})
        parity = {'status': 'measured', 'cpu_reference': str(reference), 'comparison_calls': len(comparisons),
                  'maximum_absolute_probability_difference': max(r['max_absolute_probability_difference'] for r in comparisons),
                  'native_choices_equal': sum(r['native_choice_equal'] for r in comparisons),
                  'comparisons': comparisons,
                  'note': 'Observed numerical device comparison. No weights, temperature or prompt adjusted to force agreement.'}
        save_json(output / 'cpu_gpu_comparison.json', parity)
        print(json.dumps({k: v for k, v in parity.items() if k != 'comparisons'}), flush=True)
    records = []
    fields = ['participant_id', 'block', 'game', 'trial_num', 'human_choice',
              'sdk_raw_probability_H', 'sdk_raw_probability_I', 'legal_probability_H', 'legal_probability_I',
              'legal_log_likelihood_eps1e8', 'native_choice', 'input_tokens', 'inference_seconds']
    with (output / 'trials.csv').open('w', newline='') as csv_file, (output / 'requests.jsonl').open('w') as request_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for (participant, block, game), game_df in data.groupby(['participant_id', 'block', 'game'], sort=True):
            game_df = game_df.sort_values('trial')
            intro = helpers['build_game_intro'](game_df, game)
            forced = helpers['format_forced_trials'](game_df[game_df.type.eq('forced')])
            for _, row in game_df[game_df.type.eq('free')].iterrows():
                past = helpers['format_past_trials'](game_df[game_df.type.eq('free') & game_df.trial.lt(row.trial)])
                prompt = str(intro + forced + past) + 'You press <<'
                t0 = time.monotonic()
                p = model.choice_probabilities(prompt, question, ['H', 'I'])
                seconds = time.monotonic() - t0
                actual = {'H': 0, 'I': 1}[row.choice]
                record = {'participant_id': int(participant), 'block': int(block), 'game': int(game),
                          'trial_num': int(row.trial), 'human_choice': row.choice,
                          'sdk_raw_probability_H': model.last_raw_probabilities[0],
                          'sdk_raw_probability_I': model.last_raw_probabilities[1],
                          'legal_probability_H': float(p[0]), 'legal_probability_I': float(p[1]),
                          'legal_log_likelihood_eps1e8': float(np.log(p[actual] + 1e-8)),
                          'native_choice': model.last_native_choice, 'input_tokens': model.last_input_tokens,
                          'inference_seconds': seconds}
                writer.writerow(record)
                request_file.write(json.dumps({
                    'participant_id': int(participant), 'block': int(block), 'game': int(game),
                    'trial_num': int(row.trial), 'state': prompt, 'question': question,
                }, ensure_ascii=False) + '\n')
                records.append(record)
                if len(records) % 32 == 0:
                    csv_file.flush()
                    request_file.flush()
                    progress = {'status': 'running', 'completed_free_choices': len(records),
                                'expected_free_choices': 6720, 'elapsed_seconds': time.monotonic() - started}
                    save_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)

    frame = pd.DataFrame(records)
    if len(frame) != 6720:
        raise ValueError(f'Incomplete run: {len(frame)} free choices.')
    observed = np.where(frame.human_choice.eq('H'), frame.legal_probability_H, frame.legal_probability_I)
    summary = {
        'status': 'complete', 'source_status': 'new_run_typed_choice_adapter',
        'n_participants': frame.participant_id.nunique(), 'n_free_choices': len(frame),
        'legal_conditioned_nll_eps1e8': float(-frame.legal_log_likelihood_eps1e8.mean()),
        'original_full_vocabulary_nll': None, 'epsilon': 1e-8,
        'zero_target_probability_count': int((observed == 0).sum()),
        'participant_legal_nll': {str(int(k)): float(-v) for k, v in
                                  frame.groupby('participant_id').legal_log_likelihood_eps1e8.mean().items()},
        'maximum_input_tokens': int(frame.input_tokens.max()),
        'total_seconds': time.monotonic() - started, 'load_seconds': load_seconds,
        'mean_inference_seconds': float(frame.inference_seconds.mean()),
        'device': args.device,
        'gpu_hours_wall_time_shared': ((time.monotonic() - started) / 3600 if args.device.startswith('cuda') else 0),
        'timing_context': config['timing_context'],
        'notes': 'Complete original test and intact original Centaur prompts; typed-choice head and legal NLL are an explicit interface adaptation.',
    }
    save_json(output / 'summary.json', summary)
    save_json(output / 'progress.json', summary)
    config['status'] = 'complete'
    save_json(output / 'config.json', config)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
