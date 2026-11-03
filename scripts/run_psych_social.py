#!/usr/bin/env python3
"""Run full public social-task materials with frozen scalar model interfaces."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.psych_social_tasks import TASKS, VERSION, load_trials, inventory, score, summarize
from scripts.run_psych_typed import predict_original_scalar, write_json


def run(args):
    trials = load_trials(args.task)
    info = inventory(trials)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        from src.psych_typed import audit_lengths
        from src.social_numeric_models import audit_numeric_lengths
        audit = audit_numeric_lengths if args.task == 'nonliteral_number_interpretation' else audit_lengths
        info['model_input_lengths'] = audit(trials, args.inspect_models)
        info['input_audit_status'] = 'complete'
        write_json(args.output / 'source_inventory.json', info)
        with (args.output / 'response_manifest.jsonl').open('w') as stream:
            for trial in trials:
                stream.write(json.dumps(trial.identifiers(), ensure_ascii=False) + '\n')
        print(json.dumps(info), flush=True)
        return
    if not args.model_label:
        raise ValueError('--model-label is required for inference.')
    numeric = args.task == 'nonliteral_number_interpretation'
    if not numeric and not args.adapter_file:
        raise ValueError('Choice inference requires --adapter-file.')
    config = {'task': args.task, 'protocol_version': VERSION, 'model_label': args.model_label,
              'adapter_file': str(args.adapter_file) if args.adapter_file else None,
              'adapter_source': args.adapter_file.read_text() if args.adapter_file else (ROOT / 'src/social_numeric_models.py').read_text(),
              'device': args.device, 'responses': len(trials), 'batch_size': 1}
    path = args.output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Resume requires the exact same configuration.')
    write_json(path, config)
    write_json(args.output / 'source_inventory.json', info)
    batches = args.output / 'batches'
    batches.mkdir(exist_ok=True)
    rows, pending = [], []
    for i, trial in enumerate(trials):
        path = batches / f'{i:07d}.json'
        if path.exists():
            saved = json.loads(path.read_text())['records']
            if len(saved) != 1 or saved[0]['id'] != trial.id:
                raise ValueError('Resume response ID differs from the original input.')
            rows.extend(saved)
        else:
            pending.append((i, trial, path))
    try:
        if pending:
            if numeric:
                from src.social_numeric_models import NumericSocialModel
                agent = NumericSocialModel(args.model_label, args.device)
            else:
                from src.paper_horizon import load_agent
                agent = load_agent(args.adapter_file, args.device)
            write_json(args.output / 'model_metadata.json', getattr(agent, 'metadata', {}))
            for i, trial, path in pending:
                if numeric:
                    prediction = agent.predict(trial)
                else:
                    prediction = predict_original_scalar(agent, trial, args.device)
                row = score(trial, prediction)
                row['runner_batch_id'] = i
                write_json(path, {'begin': i, 'records': [row]})
                rows.append(row)
                progress = {'status': 'running', 'completed_responses': len(rows), 'expected_responses': len(trials),
                            'updated_utc': datetime.now(timezone.utc).isoformat()}
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    except Exception:
        failure = {'utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(),
                   'completed_responses': len(rows), 'expected_responses': len(trials)}
        with (args.output / 'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure) + '\n')
        write_json(args.output / 'summary.json', summarize(rows, len(trials)))
        write_json(args.output / 'progress.json', {'status': 'failed', **failure})
        raise
    result = summarize(rows, len(trials))
    elapsed = sum(x['batch_elapsed_seconds'] for x in rows)
    result.update(model_label=args.model_label, task=args.task, completed_batch_wall_seconds=elapsed,
                  shared_responses_per_second=len(rows) / elapsed if elapsed else None)
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'progress.json', {'status': 'complete', 'completed_responses': len(rows), 'expected_responses': len(trials)})
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=list(TASKS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter-file', type=Path)
    p.add_argument('--model-label', choices=['laya', 'intern_decision_4b', 'llama8b', 'minitaur8b', 'qwen35_4b'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--inspect-models', nargs='*', default=[])
    run(p.parse_args())


if __name__ == '__main__':
    main()
