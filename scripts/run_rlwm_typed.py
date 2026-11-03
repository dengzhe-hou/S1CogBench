#!/usr/bin/env python3
"""Run all author-eligible CF12 responses with declared complete block histories."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rlwm_typed import (PROTOCOL_VERSION, DEFAULT_SOURCE, read_source, build_trials,
                            inventory, score, summarize, save_source_rows)
from src.psych_typed import audit_lengths
from scripts.run_psych_typed import predict_original_scalar, write_json


def completed(output, rows, config):
    rows.sort(key=lambda r: (r['source_row'], r['trial_index']))
    result = summarize(rows, config['responses'])
    seconds = sum(r['batch_elapsed_seconds'] for r in rows)
    result.update(task=config['task'], model_label=config['model_label'],
                  completed_batch_wall_seconds=seconds,
                  shared_responses_per_second=len(rows) / seconds if seconds else None)
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'status': 'complete', 'completed_responses': len(rows),
        'expected_responses': config['responses'], 'updated_utc': datetime.now(timezone.utc).isoformat()})
    return result


def run(args):
    original_rows = read_source(args.source)
    trials = build_trials(original_rows)
    source_inventory = inventory(original_rows, trials)
    args.output.mkdir(parents=True, exist_ok=True)
    save_source_rows(args.output / 'source_rows.jsonl', original_rows)
    if args.inspect_only:
        source_inventory['model_input_lengths'] = audit_lengths(trials, args.inspect_models)
        write_json(args.output / 'source_inventory.json', source_inventory)
        print(json.dumps(source_inventory, ensure_ascii=False), flush=True)
        return
    if args.adapter_file is None or args.model_label is None:
        raise ValueError('Inference needs --adapter-file and --model-label.')
    config = {'protocol_version': PROTOCOL_VERSION, 'task': args.task,
        'source': str(args.source.resolve()), 'adapter_file': str(args.adapter_file.resolve()),
        'adapter_source': args.adapter_file.read_text(), 'model_label': args.model_label,
        'device': args.device, 'batch_size': 1, 'responses': len(trials),
        'history': source_inventory['history'], 'split': source_inventory['split'],
        'execution': 'original_adapter.choice_probabilities'}
    config_file = args.output / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError('Resume requires identical configuration; use a new directory for changes.')
    else:
        write_json(config_file, config)
        write_json(args.output / 'source_inventory.json', source_inventory)
    batches = args.output / 'batches'
    batches.mkdir(exist_ok=True)
    rows, pending = [], []
    for begin, trial in enumerate(trials):
        path = batches / f'{begin:07d}.json'
        if path.exists():
            data = json.loads(path.read_text())['records']
            if len(data) != 1 or data[0]['id'] != trial.id:
                raise ValueError('Completed response ID does not match the original source row.')
            rows.extend(data)
        else:
            pending.append((begin, trial, path))
    if not pending:
        completed(args.output, rows, config)
        return
    from src.paper_horizon import load_agent
    started = time.monotonic()
    try:
        agent = load_agent(args.adapter_file, args.device)
        metadata = getattr(agent, 'metadata', {})
        metadata_file = args.output / 'model_metadata.json'
        if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
            raise ValueError('Actual model metadata changed; keep old results and use a new run directory.')
        if not metadata_file.exists():
            write_json(metadata_file, metadata)
        for begin, trial, path in pending:
            prediction = predict_original_scalar(agent, trial, args.device)
            row = score(trial, prediction)
            row['runner_batch_id'] = begin
            write_json(path, {'begin': begin, 'records': [row]})
            rows.append(row)
            write_json(args.output / 'progress.json', {'status': 'running',
                'completed_responses': len(rows), 'expected_responses': len(trials),
                'attempt_seconds': time.monotonic() - started,
                'updated_utc': datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'task': args.task, 'completed_responses': len(rows),
                              'expected_responses': len(trials)}), flush=True)
    except Exception:
        failure = {'utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(),
                   'completed_responses': len(rows), 'expected_responses': len(trials)}
        with (args.output / 'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure) + '\n')
        write_json(args.output / 'summary.json', summarize(rows, len(trials)))
        write_json(args.output / 'progress.json', {'status': 'failed', **failure})
        raise
    print(json.dumps(completed(args.output, rows, config), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['rlwm'], default='rlwm')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--adapter-file', type=Path)
    parser.add_argument('--model-label')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, choices=[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--inspect-models', nargs='*', default=[],
                        choices=['laya', 'intern_decision_4b', 'llama8b', 'minitaur8b', 'qwen35_4b'])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
