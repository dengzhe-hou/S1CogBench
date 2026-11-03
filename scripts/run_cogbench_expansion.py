"""Run complete original CogBench task settings through frozen choice adapters."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.cogbench_expansion import DEFAULT_RUNS, ChoiceLLM, run_one, score_runs, summary_metrics
from src.paper_horizon import load_agent, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=DEFAULT_RUNS)
    parser.add_argument('--adapter-file', type=Path)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--runs', type=int, help='Explicit source protocol; default is original code count.')
    parser.add_argument('--version', choices=['1', '2'], default='1')
    parser.add_argument('--base-pay', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--action-rule', choices=['native', 'sample'], default='native')
    parser.add_argument('--inference-only', action='store_true', help='Leave original CPU scores pending.')
    parser.add_argument('--score-only', action='store_true', help='Score an existing complete trials.csv on CPU.')
    args = parser.parse_args()
    if args.score_only:
        config = json.loads((args.output / 'config.json').read_text())
        if config['task'] != args.task or config['model_label'] != args.model_label:
            parser.error('Scoring task/model must match saved run configuration.')
        frame = pd.read_csv(args.output / 'trials.csv')
        summary = json.loads((args.output / 'summary.json').read_text())
        started = time.perf_counter()
        write_json(args.output / 'status.json', {**summary, 'status': 'scoring_running'})
        try:
            scores = score_runs(args.task, frame, args.model_label, config['seed'])
            scores.to_csv(args.output / 'official_scores.csv', index=False)
            summary.update(status='complete', original_scoring='complete',
                           official_scores_file='official_scores.csv',
                           original_scoring_seconds=time.perf_counter() - started,
                           original_scoring_seed=config['seed'])
            write_json(args.output / 'summary.json', summary)
            write_json(args.output / 'status.json', summary)
        except Exception as exc:
            write_json(args.output / 'status.json', {**summary, 'status': 'scoring_failed',
                                                     'error': repr(exc), 'traceback': traceback.format_exc()})
            raise
        return
    if args.adapter_file is None:
        parser.error('--adapter-file is required for inference.')
    if args.version != '1' and args.task != 'bart':
        parser.error('Only BART has a separately implemented v2 in these sources.')
    n_runs = args.runs if args.runs is not None else DEFAULT_RUNS[args.task]
    if n_runs < 1:
        parser.error('--runs must be positive.')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'runs': n_runs, 'output': str(args.output),
              'adapter_file': str(args.adapter_file.resolve()),
              'task_protocol': 'Archived official task; typed-choice interface adaptation.',
              'history': 'Full original visible history; no truncation or summarization.',
              'data_kind': 'Original fixed questionnaire / program-generated environment, not human data.',
              'timing_context': 'Shared host/device; engineering latency, not human reaction time.',
              'instrumental_fix': 'One generate and one row per effective trial; original 96 trials preserved.',
              'scope': 'Code-default runs unless explicit; not automatically the paper or archived model scale.'}
    write_json(args.output / 'config.json', config)
    started = time.perf_counter()
    all_rows = []
    try:
        agent = load_agent(args.adapter_file, args.device)
        write_json(args.output / 'model_metadata.json', getattr(agent, 'metadata', {}))
        with (args.output / 'requests.jsonl').open('w') as stream:
            bridge = ChoiceLLM(agent, args.task, stream, args.action_rule, args.device)
            for run in range(n_runs):
                frame = run_one(args.task, bridge, run, args.seed + run, args.version, args.base_pay)
                frame.to_csv(args.output / f'run_{run:04d}.csv', index=False)
                all_rows.append(frame)
                write_json(args.output / 'status.json', {'status': 'running', 'completed_runs': run + 1,
                                                         'expected_runs': n_runs, 'calls': len(bridge.records)})
        frame = pd.concat(all_rows, ignore_index=True)
        frame.to_csv(args.output / 'trials.csv', index=False)
        summary = summary_metrics(args.task, frame, bridge.records)
        summary.update(status='inference_complete_scoring_pending', original_scoring='pending',
                       task=args.task, model_label=args.model_label, total_seconds=time.perf_counter() - started,
                       timing_context=config['timing_context'])
        write_json(args.output / 'summary.json', summary)
        if not args.inference_only:
            scoring_started = time.perf_counter()
            scores = score_runs(args.task, frame, args.model_label, args.seed)
            scores.to_csv(args.output / 'official_scores.csv', index=False)
            summary.update(status='complete', original_scoring='complete',
                           official_scores_file='official_scores.csv',
                           original_scoring_seconds=time.perf_counter() - scoring_started,
                           original_scoring_seed=args.seed,
                           total_seconds=time.perf_counter() - started)
            write_json(args.output / 'summary.json', summary)
        write_json(args.output / 'status.json', summary)
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        write_json(args.output / 'status.json', {'status': 'failed', 'completed_runs': len(all_rows),
                                                 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
