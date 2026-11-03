#!/usr/bin/env python3
"""Run complete Medin/weather Psych-101 held-out responses without changing SHJ."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.psych_category_tasks import (PROTOCOL_VERSION, inventory, load_category_trials,
                                      score, summarize_category)
from src.psych_typed import DEFAULT_SOURCE, audit_lengths
from scripts.run_psych_typed import predict_original_scalar, write_json


def completed(output, rows, config):
    rows.sort(key=lambda r: (r['source_row'], r['trial_index']))
    result = summarize_category(rows, config['responses'])
    durations = {(r['runner_batch_id'], r['batch_id']): r['batch_elapsed_seconds'] for r in rows}
    seconds = sum(durations.values())
    result.update(task=config['task'], model_label=config['model_label'],
                  completed_batch_wall_seconds=seconds,
                  shared_responses_per_second=len(rows) / seconds if seconds else None)
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'status': 'complete', 'completed_responses': len(rows),
        'expected_responses': config['responses'], 'updated_utc': datetime.now(timezone.utc).isoformat()})
    return result


def run(args):
    trials = load_category_trials(args.task, args.source)
    source_inventory = inventory(trials)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        source_inventory['model_input_lengths'] = audit_lengths(trials, args.inspect_models)
        write_json(args.output / 'source_inventory.json', source_inventory)
        print(json.dumps(source_inventory, ensure_ascii=False), flush=True)
        return
    if args.adapter_file is None or args.model_label is None or args.batch_size < 1:
        raise ValueError('Inference needs --adapter-file, --model-label and positive --batch-size.')
    config = {'protocol_version': PROTOCOL_VERSION, 'task': args.task,
        'source': str(args.source.resolve()), 'adapter_file': str(args.adapter_file.resolve()),
        'adapter_source': args.adapter_file.read_text(), 'model_label': args.model_label,
        'device': args.device, 'batch_size': args.batch_size, 'responses': len(trials),
        'history': source_inventory['history'], 'split': source_inventory['split'],
        'execution': 'original_adapter.choice_probabilities' if args.batch_size == 1 else 'shared_batched_choice_interface'}
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
    for begin in range(0, len(trials), args.batch_size):
        group = trials[begin:begin + args.batch_size]
        path = batches / f'{begin:07d}.json'
        if path.exists():
            data = json.loads(path.read_text())['records']
            if [r['id'] for r in data] != [t.id for t in group]:
                raise ValueError('Completed batch IDs do not match the original source rows.')
            rows.extend(data)
        else:
            pending.append((begin, group, path))
    if not pending:
        completed(args.output, rows, config)
        return
    from src.paper_horizon import load_agent
    from src.batched_choices import ChoiceRequest, choice_probabilities_batch
    started = time.monotonic()
    try:
        agent = load_agent(args.adapter_file, args.device)
        metadata = getattr(agent, 'metadata', {})
        metadata_file = args.output / 'model_metadata.json'
        if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
            raise ValueError('Actual model metadata changed; keep old results and use a new run directory.')
        if not metadata_file.exists():
            write_json(metadata_file, metadata)
        for begin, group, path in pending:
            if args.batch_size == 1:
                outputs = [predict_original_scalar(agent, group[0], args.device)]
            else:
                requests = [ChoiceRequest(t.state, t.question, list(t.output_keys)) for t in group]
                outputs = choice_probabilities_batch(agent, requests, batch_size=args.batch_size)
            if len(outputs) != len(group):
                raise ValueError('Missing model responses in this batch.')
            scored = [score(t, output) for t, output in zip(group, outputs)]
            for row in scored:
                row['runner_batch_id'] = begin
            write_json(path, {'begin': begin, 'records': scored})
            rows.extend(scored)
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
        write_json(args.output / 'summary.json', summarize_category(rows, len(trials)))
        write_json(args.output / 'progress.json', {'status': 'failed', **failure})
        raise
    print(json.dumps(completed(args.output, rows, config), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['medin', 'weather'], required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--adapter-file', type=Path)
    parser.add_argument('--model-label')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--inspect-models', nargs='*', default=[],
                        choices=['laya', 'intern_decision_4b', 'llama8b', 'minitaur8b', 'qwen35_4b'])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
