#!/usr/bin/env python3
"""Run the approved Intern integer protocol with complete per-response evidence.

The entire original source remains in scope. This runner adds typed-response
likelihood; it does not reproduce Centaur's full-vocabulary likelihood. The
Intern checkpoint stays frozen BF16 with the original 8192-token input budget.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.four_task_interfaces import iter_probability_sessions, iter_assessment_sessions, score_response, PROTOCOL_VERSION
from scripts.run_psych_typed import predict_original_scalar as predict_scalar, write_json
from examples.intern_decision_integer_sequence_agent_20261019 import PROTOCOL


def predict_record(agent, trial, device):
    prediction = predict_scalar(agent, trial, device)
    row = score_response(trial, prediction)
    row.update(readout_protocol=PROTOCOL,
               readout_interface=agent.last_readout_interface,
               prompt_characters=len(agent.last_prompt),
               history_characters=len(trial.state),
               history_response_count=trial.trial_index,
               latency_seconds_shared_gpu=prediction['batch_elapsed_seconds'])
    for field in ('sequence_logprobabilities', 'allowed_sequence_mass',
                  'max_reply_tokens', 'integer_token_sequences'):
        if hasattr(agent, 'last_' + field):
            row[field] = getattr(agent, 'last_' + field)
    return row


def sessions(task):
    if task == 'probability_coherence_judgment':
        yield from iter_probability_sessions()
    else:
        for domain in ('grammar', 'logic'):
            yield from iter_assessment_sessions(domain)


def run(args):
    expected = 2460 if args.task == 'probability_coherence_judgment' else 108248
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'task': args.task, 'protocol_version': PROTOCOL_VERSION,
              'readout_protocol': PROTOCOL, 'sequence_evidence': 'Complete joint log probabilities, raw joint probabilities, legal mass, reply token sequences, prompt/history lengths and shared-device latency.',
              'responses': expected, 'adapter_file': str(args.adapter_file.resolve()),
              'adapter_source': args.adapter_file.read_text(), 'device': args.device,
              'model_label': args.model_label, 'history': 'All original human responses; no model rollout',
              'self_assessment_domain_scope': ['grammar', 'logic'] if args.task == 'answer_self_assessment' else None,
              'difficulty_scale': 'Published Psych transcript 0–10 retained; original QSF 1–10 discrepancy disclosed',
              'original_full_vocabulary_nll': 'Not produced by this typed-response runner'}
    config_path = args.output / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume requires the same configuration.'
    else:
        write_json(config_path, config)
    output = args.output / 'responses.jsonl'
    seen, sums = set(), defaultdict(Counter)

    def accumulate(row):
        assert row['id'] not in seen, row['id']
        seen.add(row['id'])
        for key in ('all', row['experiment'], row['experiment'] + ':' + row['response_kind'], row['readout_interface']):
            s = sums[key]
            s['responses'] += 1
            s['latency_seconds_shared_gpu'] += row['latency_seconds_shared_gpu']
            s['nll_sum_finite'] += row['typed_choice_nll'] or 0
            s['nll_infinite_count'] += int(row['typed_choice_nll_is_infinite'])
            s['matches_human'] += row['matches_human_choice']
            if row.get('native_scale_absolute_error') is not None:
                s['numeric_rows'] += 1
                s['absolute_error_sum'] += row['native_scale_absolute_error']
            if row.get('category_correct_under_human_history') is not None:
                s['objective_rows'] += 1
                s['objective_correct'] += row['category_correct_under_human_history']

    if output.exists():
        for line in output.open():
            accumulate(json.loads(line))

    def summary(status):
        result = {}
        for key, s in sums.items():
            result[key] = dict(s, mean_typed_nll=None if s['nll_infinite_count'] else s['nll_sum_finite']/s['responses'],
                mean_typed_nll_is_infinite=bool(s['nll_infinite_count']),
                mean_numeric_absolute_error=s['absolute_error_sum']/s['numeric_rows'] if s['numeric_rows'] else None,
                objective_accuracy=s['objective_correct']/s['objective_rows'] if s['objective_rows'] else None)
        return {'status': status, 'responses': len(seen), 'expected_responses': expected,
                'groups': result, 'full_scope_complete': len(seen) == expected,
                'self_assessment_interpretation': 'Estimate about the supplied complete human history, not the model own answer sequence.',
                'numeric_error_interpretation': 'All/experiment/readout aggregates mix numeric units and are not a unified metric; formal numeric comparisons use experiment:response_kind groups.',
                'original_protocol_nll': None, 'shared_device_timing': True}

    agent = None
    try:
        with output.open('a') as stream:
            expected_ids = set()
            for trials, metadata in sessions(args.task):
                for trial in trials:
                    expected_ids.add(trial.id)
                    if trial.id in seen:
                        continue
                    if agent is None:
                        from src.paper_horizon import load_agent
                        agent = load_agent(args.adapter_file, args.device)
                        write_json(args.output / 'model_metadata.json', agent.metadata)
                    row = predict_record(agent, trial, args.device)
                    if 'raw_score' in metadata:
                        row['human_twenty_question_score_for_scoring_only'] = metadata['raw_score']
                        if trial.response_kind == 'absAssess':
                            row['human_history_score_estimate_error'] = row['reported_integer'] - metadata['raw_score']
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    accumulate(row)
                write_json(args.output / 'progress.json', {'status': 'running', 'completed_responses': len(seen),
                    'expected_responses': expected, 'updated_utc': datetime.now(timezone.utc).isoformat()})
                print(json.dumps({'completed_responses': len(seen), 'expected_responses': expected}), flush=True)
            assert len(expected_ids) == expected and seen == expected_ids
    except Exception:
        failure = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(),
                   'completed_responses': len(seen), 'expected_responses': expected}
        with (args.output / 'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure) + '\n')
        write_json(args.output / 'summary.json', summary('failed'))
        write_json(args.output / 'progress.json', {'status': 'failed', **failure})
        raise
    write_json(args.output / 'summary.json', summary('complete'))
    write_json(args.output / 'progress.json', {'status': 'complete', 'completed_responses': len(seen),
               'expected_responses': expected, 'updated_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['probability_coherence_judgment', 'answer_self_assessment'])
    parser.add_argument('--adapter-file', type=Path, default=ROOT / 'examples/intern_decision_integer_sequence_agent_20261019.py')
    parser.add_argument('--model-label', default='intern_decision_4b_complete_integer_sequence')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
