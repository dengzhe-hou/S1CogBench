#!/usr/bin/env python3
"""Run complete original SHJ test responses through the shared typed interface."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.psych_typed import (DEFAULT_SOURCE, PROTOCOL_VERSION, audit_lengths,
                            load_shj_trials, score_prediction, source_inventory, summarize)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_completed(output, records, config):
    records.sort(key=lambda r: (r['source_row'], r['trial_index']))
    summary = summarize(records, config['responses'])
    batch_times = {}
    for row in records:
        if row.get('batch_elapsed_seconds') is not None:
            batch_times[(row['runner_batch_id'], row['batch_id'])] = row['batch_elapsed_seconds']
    summary.update(model_label=config['model_label'], batch_size=config['batch_size'],
                   completed_batch_wall_seconds=sum(batch_times.values()),
                   source_inventory='source_inventory.json', model_metadata='model_metadata.json')
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', {
        'status': 'complete', 'completed_responses': len(records),
        'expected_responses': config['responses'],
        'updated_utc': datetime.now(timezone.utc).isoformat()})
    return summary


def predict_original_scalar(adapter, trial, device):
    """Batch size one uses the existing adapter unchanged, without padded batching."""
    import torch
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    started = time.monotonic()
    probabilities = adapter.choice_probabilities(trial.state, trial.question, list(trial.output_keys))
    if cuda:
        torch.cuda.synchronize(device)
    prediction = {'probabilities': [float(p) for p in probabilities],
                  'native_choice': getattr(adapter, 'last_native_choice', None),
                  'batch_id': 0, 'batch_size': 1,
                  'batch_elapsed_seconds': time.monotonic() - started,
                  'inference_interface': 'original_scalar_choice_probabilities'}
    for name in ['raw_probabilities', 'input_tokens', 'allowed_token_mass', 'label_mapping']:
        if hasattr(adapter, 'last_' + name):
            value = getattr(adapter, 'last_' + name)
            if name == 'raw_probabilities' and value is not None:
                value = [float(p) for p in value]
            prediction[name] = value
    return prediction


def run(args):
    trials = load_shj_trials(args.source)
    inventory = source_inventory(trials)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        inventory['model_input_lengths'] = audit_lengths(trials, args.inspect_models)
        write_json(output / 'source_inventory.json', inventory)
        print(json.dumps(inventory, ensure_ascii=False), flush=True)
        return
    if not args.adapter_file or not args.model_label:
        raise ValueError('--adapter-file and --model-label are required for inference.')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive.')
    config = {'protocol_version': PROTOCOL_VERSION, 'source': str(args.source.resolve()),
              'adapter_file': str(args.adapter_file.resolve()),
              'adapter_source': args.adapter_file.read_text(),
              'model_label': args.model_label, 'device': args.device,
              'batch_size': args.batch_size, 'responses': len(trials),
              'history': inventory['history'], 'split': inventory['split']}
    if args.batch_size == 1:
        config['scalar_execution'] = 'original_adapter.choice_probabilities'
    config_file = output / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError('Resume requires the same config; use a new output directory for a changed run.')
    else:
        write_json(config_file, config)
        write_json(output / 'source_inventory.json', inventory)
    batches_dir = output / 'batches'
    batches_dir.mkdir(exist_ok=True)
    records, pending = [], []
    for begin in range(0, len(trials), args.batch_size):
        batch_trials = trials[begin:begin + args.batch_size]
        path = batches_dir / f'{begin:07d}.json'
        if path.exists():
            data = json.loads(path.read_text())
            if [r['id'] for r in data['records']] != [t.id for t in batch_trials]:
                raise ValueError('Completed batch IDs differ from the original ordered test rows.')
            records.extend(data['records'])
        else:
            pending.append((begin, batch_trials, path))
    if not pending:
        write_completed(output, records, config)
        print(json.dumps({'status': 'already_complete', 'responses': len(records)}), flush=True)
        return
    from src.paper_horizon import load_agent
    from src.batched_choices import ChoiceRequest, choice_probabilities_batch
    started = time.monotonic()
    try:
        adapter = load_agent(args.adapter_file, args.device)
        metadata = getattr(adapter, 'metadata', {})
        metadata_file = output / 'model_metadata.json'
        if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
            raise ValueError('Actual model metadata changed; retain previous run and use a new output directory.')
        if not metadata_file.exists():
            write_json(metadata_file, metadata)
        for begin, batch_trials, path in pending:
            requests = [ChoiceRequest(t.state, t.question, list(t.output_keys)) for t in batch_trials]
            if args.batch_size == 1:
                predictions = [predict_original_scalar(adapter, batch_trials[0], args.device)]
            else:
                predictions = choice_probabilities_batch(adapter, requests, batch_size=args.batch_size)
            if len(predictions) != len(batch_trials):
                raise ValueError('Adapter returned a different number of predictions than requests.')
            scored = [score_prediction(t, p) for t, p in zip(batch_trials, predictions)]
            # Local adapter batch IDs may reset each call. Record the source-order
            # ID so shared batch time is counted exactly once in later analyses.
            for row in scored:
                row['runner_batch_id'] = begin
            write_json(path, {'begin': begin, 'records': scored})
            records.extend(scored)
            write_json(output / 'progress.json', {
                'status': 'running', 'completed_responses': len(records),
                'expected_responses': len(trials), 'attempt_seconds': time.monotonic() - started,
                'updated_utc': datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'completed_responses': len(records), 'expected_responses': len(trials),
                              'batch_begin': begin}), flush=True)
    except Exception:
        failure = {'utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(),
                   'completed_responses': len(records), 'expected_responses': len(trials)}
        with (output / 'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure, ensure_ascii=False) + '\n')
        write_json(output / 'summary.json', summarize(records, len(trials)))
        write_json(output / 'progress.json', {
            'status': 'failed', 'completed_responses': len(records),
            'expected_responses': len(trials), 'updated_utc': failure['utc']})
        raise
    summary = write_completed(output, records, config)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--adapter-file', type=Path)
    parser.add_argument('--model-label')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true', help='Audit all source rows on CPU; no model inference.')
    parser.add_argument('--inspect-models', nargs='*', default=[],
                        choices=['laya', 'llama8b', 'minitaur8b', 'intern_decision_4b', 'qwen35_4b'])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
