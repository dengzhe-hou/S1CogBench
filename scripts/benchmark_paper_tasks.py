#!/usr/bin/env python3
"""Export/score published tasks and run external choice/event model adapters."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = {
    'horizon-predictive': {
        'paper': 'https://arxiv.org/abs/2508.07887',
        'task_family': 'exploration_and_adaptation',
        'data_kind': 'published_human_behavior',
        'participants': 6, 'decisions': 6720,
        'split': 'Original six-person test; 25 training participants kept separate.',
        'metric': 'legal_conditioned_nll_eps1e8', 'epsilon': 1e-8,
        'score_status': 'explicit_legal_probability_adaptation',
        'original_metric': 'Full-vocabulary NLL is preserved in original model runs and reported separately.',
        'history': 'Original per-game prompt; previous human choices and observed rewards.',
        'source': 'external/paper_matched_centaur_evaluation/horizon/data/in/test_data.csv',
        'task_state': 'full_data_export_and_prediction_scoring_available',
    },
    'mouselab-predictive': {
        'paper': 'https://doi.org/10.1038/s41562-022-01332-8',
        'task_family': 'information_acquisition_and_stopping',
        'data_kind': 'published_human_behavior',
        'participants': 95, 'decisions': 20363,
        'split': 'Original five folds within each participant; whole trials kept together.',
        'metric': 'unsmoothed_action_nll', 'epsilon': 0.0,
        'score_status': 'original_formula_with_explicit_text_interface',
        'history': 'Complete revealed-node prefix in current trial; reset between trials.',
        'source': 'data/mouselab_callaway2022_official/trials.json',
        'task_state': 'full_data_export_and_prediction_scoring_available',
    },
}
POSTERIOR_TASKS = {
    'probabilistic-posterior': {
        'paper': 'https://proceedings.mlr.press/v235/coda-forno24a.html',
        'task_family': 'evidence_integration_and_planning',
        'data_kind': 'new_program_generated_urn_observations_not_human_data',
        'version': '1', 'runs_code_default': 200, 'queries_code_default': 400,
        'runs_paper_statement': 100, 'queries_paper_scale': 200,
        'scale_protocol': 'Keep all 200 code-default runs; predeclare first 100 as paper-scale report.',
        'output': 'reported_event_probability = P(named target urn | visible prior and observed color)',
        'metric': 'current_official_strict_q95_posterior_error_and_prior_likelihood_weights',
        'additional_metrics': 'Full posterior MAE, generated-event Brier/NLL, separately rounded report.',
        'source': 'external/CogBench/Experiments/ProbabilisticReasoning/query.py',
        'score_status': 'original_generator_and_current_scorer_with_explicit_event_probability_adapter',
        'protocol_deviations': 'New replay seeds and typed event interface; publication/code scale and '
            'regression-intercept differences are disclosed, not silently reconciled.',
        'task_state': 'full_static_export_scoring_and_external_callback_available',
        'adapter_contract': 'create_agent(device) returns an object with '
            'posterior_probability(state, question); return a scalar probability in [0,1].',
    },
}
GENERATIVE_TASKS = {
    'mouselab-generative': {
        'paper': 'https://doi.org/10.1038/s41562-022-01332-8',
        'task_family': 'information_acquisition_and_stopping',
        'data_kind': 'new_program_generated_reward_trees_not_human_trajectories',
        'participant_mdp_slots': 95, 'trials_per_slot': 25, 'repeats_per_trial': 10,
        'episodes': 23750, 'environment_cost': 0,
        'protocol': 'Original Exp1 meta-MDP, legal reveal frontier, own observations and rollout(policy, s).',
        'metric': 'original_STOP_best_path_expected_reward_and_reveal_count',
        'score_status': 'original_environment_with_explicit_text_choice_and_seeded_replay_adaptations',
        'source': 'external/callaway2022_archive/planning/model',
        'task_state': 'full_closed_loop_external_callback_available',
        'action_rules': ['native', 'sample'],
        'adapter_contract': 'Trusted local file defines create_agent(device); returned object implements '
            'choice_probabilities(state, question, output_keys).',
        'seed_status': 'New same-tree seeded replay; original simulate.jl was unseeded. '
            'Original individual fitted-policy simulations are not paired with these trees.',
        'stopping_semantics': 'Meta-MDP automatically selects the best expected route; '
            'human subsequent route selection and realized route payoff are not measured.',
        'cost_status': 'No imposed lambda or conversion of human 3-second wait into points.',
    },
    'twostep-generative': {
        'paper': 'https://proceedings.mlr.press/v235/coda-forno24a.html',
        'task_family': 'evidence_integration_and_planning',
        'data_kind': 'new_environment_samples_from_original_public_generator',
        'runs': 100, 'trials_per_run': 20, 'decisions': 4000,
        'protocol': 'CogBench version 1; original transition, reward drift and complete own history.',
        'metric': 'original_mean_reward_and_pooled_OLS_reward_transition_stay_interaction',
        'score_status': 'original_environment_with_external_typed_choice_adapter',
        'source': 'external/CogBench/Experiments/TwoStepTask/query.py',
        'task_state': 'full_closed_loop_external_callback_available',
        'action_rules': ['native', 'sample'],
        'adapter_contract': 'Trusted local file defines create_agent(device), returning an agent with '
            'choice_probabilities(state, question, output_keys).',
        'seed_status': 'New explicitly seeded replay protocol; original CogBench code did not fix seeds.',
        'inference_status': 'Typed-choice interface is an explicit adaptation, not the original LLM decoder.',
        'human_prediction': 'Not provided: published human 40 x 200 data are a different protocol.',
    },
    'horizon-generative': {
        'paper': 'https://arxiv.org/abs/2508.07887',
        'task_family': 'exploration_and_adaptation',
        'data_kind': 'original_published_generated_reward_timelines',
        'participants': 31, 'rows': 74400, 'decisions': 34720,
        'protocol': 'Complete original participant timelines, seeds, game history and action-specific rewards.',
        'metric': 'original_notebook_optimal_choice_by_horizon_and_trial',
        'score_status': 'original_environment_with_external_typed_choice_adapter',
        'source': 'external/paper_matched_centaur_evaluation/horizon/data/in/timeline_structure.csv',
        'task_state': 'full_closed_loop_external_callback_available',
        'action_rules': ['native', 'sample'],
        'native_definition': 'Use a supplied non-null last_native_choice; otherwise submitted-probability '
            'argmax, ties taking the first output_key. Record the actual source per decision.',
        'sample_definition': 'Explicit external categorical sampling; report separately from native.',
        'adapter_contract': 'Trusted local file defines create_agent(device); returned agent implements '
            'choice_probabilities(state, question, output_keys), returning normalized probabilities in key order.',
        'inference_status': 'New typed-choice interface; original LLM decoder is not reproduced.',
    },
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def new_output(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Write benchmark artifacts inside S1CogBench.')
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_horizon_records():
    import pandas as pd
    from scripts.audit_laya_paper_horizon_lengths import functions_from_source
    upstream = ROOT / 'external/paper_matched_centaur_evaluation/horizon'
    helpers = functions_from_source(upstream / 'predictive_horizon_centaur.py',
                                    ['build_game_intro', 'format_forced_trials', 'format_past_trials'])
    question = functions_from_source(ROOT / 'src/horizon.py', ['choice_question'])['choice_question'](
        labels=('H', 'I'))
    data = pd.read_csv(upstream / 'data/in/test_data.csv')
    if set(data.participant_id) != {1, 2, 7, 23, 26, 31}:
        raise ValueError('Unexpected official Horizon test participants.')
    records = []
    for (person, block, game), frame in data.groupby(['participant_id', 'block', 'game'], sort=True):
        frame = frame.sort_values('trial')
        intro = helpers['build_game_intro'](frame, game)
        forced = helpers['format_forced_trials'](frame[frame.type.eq('forced')])
        for _, row in frame[frame.type.eq('free')].iterrows():
            past = helpers['format_past_trials'](frame[frame.type.eq('free') & frame.trial.lt(row.trial)])
            records.append({
                'request': {
                    'id': f'{int(person)}/{int(block)}/{int(game)}/{int(row.trial)}',
                    'state': str(intro + forced + past) + 'You press <<',
                    'question': question, 'output_keys': ['H', 'I'],
                },
                'target': row.choice,
                'metadata': {'participant': str(int(person)), 'block': int(block),
                             'game': int(game), 'trial': int(row.trial), 'split': 'test'},
            })
    return records


def load_records(task):
    if task == 'horizon-predictive':
        records = load_horizon_records()
    else:
        from src.paper_mouselab import build_records
        records = [{
            'request': {'id': f'{r["wid"]}/{r["trial"]}/{r["decision"]}',
                        'state': r['state'], 'question': r['question'], 'output_keys': r['legal_actions']},
            'target': r['human_action'],
            'metadata': {'participant': r['wid'], 'trial': r['trial'],
                         'decision': r['decision'], 'fold': r['fold']},
        } for r in build_records()]
    spec = TASKS[task]
    if len(records) != spec['decisions'] or len({r['request']['id'] for r in records}) != len(records):
        raise ValueError('Incomplete or duplicate original task decisions.')
    if len({r['metadata']['participant'] for r in records}) != spec['participants']:
        raise ValueError('Incomplete original participant set.')
    return records


def export_task(args):
    if args.task in POSTERIOR_TASKS:
        from src.paper_probabilistic import load_records as load_posterior_records
        records = load_posterior_records()
        out = new_output(args.output)
        with (out / 'requests.jsonl').open('w') as requests, (out / 'targets.jsonl').open('w') as targets:
            for row in records:
                requests.write(json.dumps(row['request'], ensure_ascii=False, allow_nan=False) + '\n')
                targets.write(json.dumps({'id': row['request']['id'], 'target': row['target'],
                                          'metadata': row['metadata']},
                                         ensure_ascii=False, allow_nan=False) + '\n')
        write_json(out / 'task.json', {'task': args.task, **POSTERIOR_TASKS[args.task],
                                      'input_boundary': 'Send state and question only; targets are scoring-only.',
                                      'prediction_file_format': {'id': 'copy request id',
                                                                'reported_event_probability': 'scalar q'}})
        print(json.dumps({'task': args.task, 'exported_queries': len(records), 'output': str(out)}))
        return
    records = load_records(args.task)
    out = new_output(args.output)
    with (out / 'requests.jsonl').open('w') as requests, (out / 'targets.jsonl').open('w') as targets:
        for row in records:
            requests.write(json.dumps(row['request'], ensure_ascii=False, allow_nan=False) + '\n')
            targets.write(json.dumps({'id': row['request']['id'], 'human_action': row['target'],
                                      'metadata': row['metadata']}, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(out / 'task.json', {'task': args.task, **TASKS[args.task],
                                  'prediction_file_format': {'id': 'copy request id',
                                      'probabilities': 'map every output_key to its probability'},
                                  'model_input': 'state, question and output_keys only; id is an opaque join key',
                                  'targets': 'Scoring labels are separate; never include targets in a model request.'})
    print(json.dumps({'task': args.task, 'exported_decisions': len(records), 'output': str(out)}))


def probability_scores(record, predictions, epsilon):
    keys = record['request']['output_keys']
    if set(predictions) != set(keys):
        raise ValueError(f'{record["request"]["id"]}: probabilities must cover exactly the legal actions.')
    probabilities = [float(predictions[key]) for key in keys]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError('Action probabilities must be finite and lie in [0,1].')
    if abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError('Normalize probabilities over legal actions before scoring; no silent smoothing.')
    target_p = probabilities[keys.index(record['target'])]
    nll = -math.log(target_p + epsilon) if target_p + epsilon > 0 else None
    top_choice = keys[max(range(len(keys)), key=probabilities.__getitem__)]
    brier = math.fsum((p - float(key == record['target'])) ** 2
                      for key, p in zip(keys, probabilities))
    return {'target_probability': target_p, 'nll': nll, 'correct': top_choice == record['target'],
            'multiclass_brier': brier}


def aggregate(rows):
    infinite = sum(row['nll'] is None for row in rows)
    total = None if infinite else math.fsum(row['nll'] for row in rows)
    return {'decisions': len(rows), 'total_nll': total,
            'mean_nll': None if total is None else total / len(rows),
            'nll_status': 'positive_infinity' if infinite else 'finite',
            'zero_target_probability_count': sum(row['target_probability'] == 0 for row in rows),
            'mean_multiclass_brier_additional_metric': math.fsum(row['multiclass_brier'] for row in rows) / len(rows),
            'argmax_accuracy_additional_metric': sum(row['correct'] for row in rows) / len(rows)}


def score_task(args):
    if args.task in POSTERIOR_TASKS:
        from src.paper_probabilistic import load_records as load_posterior_records, score_records
        with args.predictions.open() as file:
            predictions = [json.loads(line) for line in file]
        summary = score_records(load_posterior_records(), predictions)
        out = new_output(args.output)
        write_json(out / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return
    records = load_records(args.task)
    supplied = {}
    with args.predictions.open() as file:
        for line in file:
            row = json.loads(line)
            if row['id'] in supplied:
                raise ValueError(f'Duplicate prediction id: {row["id"]}')
            supplied[row['id']] = row['probabilities']
    expected_ids = {r['request']['id'] for r in records}
    if set(supplied) != expected_ids:
        raise ValueError(f'Predictions must cover all original decisions; '
                         f'missing={len(expected_ids - supplied.keys())}, extra={len(supplied.keys() - expected_ids)}.')
    scored, participants, folds = [], defaultdict(list), defaultdict(list)
    for record in records:
        row = {'id': record['request']['id'], 'human_action': record['target'],
               **probability_scores(record, supplied[record['request']['id']], TASKS[args.task]['epsilon']),
               'metadata': record['metadata']}
        scored.append(row)
        participant = record['metadata']['participant']
        participants[participant].append(row)
        if 'fold' in record['metadata']:
            folds[(participant, record['metadata']['fold'])].append(row)
    out = new_output(args.output)
    with (out / 'decision_scores.jsonl').open('w') as file:
        for row in scored:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(out / 'participant_scores.json', {p: aggregate(rows) for p, rows in participants.items()})
    if folds:
        write_json(out / 'fold_scores.json', [{
            'participant': p, 'fold': fold, **aggregate(rows)} for (p, fold), rows in sorted(folds.items())])
    summary = {'status': 'complete', 'task': args.task, 'metric': TASKS[args.task]['metric'],
               'epsilon': TASKS[args.task]['epsilon'], 'score_status': TASKS[args.task]['score_status'],
               'participants': len(participants), 'original_cv_units': len(folds) or None,
               'predictions_source': str(args.predictions.resolve()), **aggregate(scored),
               'argmax_accuracy_definition': 'Argmax of submitted probabilities; ties take the first official output_key. '
                   'This may differ from a model SDK native choice before probability rounding.',
               'brier_definition': 'Additional action-distribution metric: sum over legal actions of '
                   '(p(a) - 1[a=human action])^2, without dividing by the number of actions. '
                   'This does not measure probability of task success.',
               'meaning': 'Human next-action prediction on complete published data; no closed-loop reward inferred.',
               'model_training_and_prompt_provenance': 'Must accompany the submitted predictions; scoring cannot infer it.'}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


def run_task(args):
    if not args.adapter_file.is_file():
        raise ValueError(f'Adapter file does not exist: {args.adapter_file}')
    if args.task in POSTERIOR_TASKS:
        from src.paper_probabilistic import run_probabilistic
        run_probabilistic(new_output(args.output), args.adapter_file, args.device)
    elif args.task == 'twostep-generative':
        from src.paper_twostep import run_twostep
        run_twostep(new_output(args.output), args.adapter_file, args.device, args.action_rule)
    elif args.task == 'mouselab-generative':
        from src.paper_mouselab_generative import run_mouselab_generative
        run_mouselab_generative(new_output(args.output), args.adapter_file, args.device, args.action_rule)
    else:
        from src.paper_horizon import run_generative
        run_generative(new_output(args.output), args.adapter_file, args.device, args.action_rule)


def predict_task(args):
    """Apply an external frozen model to every original human decision."""
    import argparse
    from src.paper_horizon import load_agent

    records = load_records(args.task)
    out = new_output(args.output)
    config = {'status': 'loading_agent', 'task': args.task, 'task_spec': TASKS[args.task],
              'adapter_file': str(args.adapter_file.resolve()), 'device': args.device,
              'decisions': len(records), 'training': 'Frozen model; no task fitting in this runner.',
              'input_boundary': 'Only state, question, ordered legal keys are sent to the model.',
              'timing_context': 'Shared device; elapsed time is not exclusive-device performance.'}
    write_json(out / 'config.json', config)
    started = time.monotonic()
    try:
        agent = load_agent(args.adapter_file, args.device)
        config.update(status='running', load_seconds=time.monotonic() - started,
                      model_metadata=getattr(agent, 'metadata', {}))
        write_json(out / 'config.json', config)
        with (out / 'predictions.jsonl').open('w') as file:
            for i, record in enumerate(records, 1):
                request = record['request']
                keys = request['output_keys']
                t0 = time.monotonic()
                p = agent.choice_probabilities(request['state'], request['question'], keys)
                probabilities = dict(zip(keys, map(float, p)))
                if len(p) != len(keys):
                    raise ValueError('Return one probability per legal action.')
                # Apply the same scoring contract before accepting an inference result.
                probability_scores(record, probabilities, TASKS[args.task]['epsilon'])
                row = {'id': request['id'], 'probabilities': probabilities,
                       'inference_seconds': time.monotonic() - t0}
                for name in ['last_native_choice', 'last_input_tokens', 'last_prompt',
                             'last_label_mapping', 'last_allowed_token_mass']:
                    if hasattr(agent, name):
                        row[name.removeprefix('last_')] = getattr(agent, name)
                file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                if i % 100 == 0 or i == len(records):
                    file.flush()
                    progress = {'status': 'running', 'completed_decisions': i,
                                'expected_decisions': len(records),
                                'elapsed_seconds': time.monotonic() - started}
                    write_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        score_task(argparse.Namespace(task=args.task, predictions=out / 'predictions.jsonl',
                                      output=out / 'scores'))
        summary = json.loads((out / 'scores/summary.json').read_text())
        summary.update(total_seconds=time.monotonic() - started, load_seconds=config['load_seconds'],
                       timing_context=config['timing_context'],
                       maximum_input_tokens=getattr(agent, 'max_input_tokens', None),
                       model_metadata=config['model_metadata'])
        import torch
        if str(args.device).startswith('cuda'):
            summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
            summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
        write_json(out / 'summary.json', summary)
        write_json(out / 'progress.json', summary)
        config['status'] = 'complete'
        write_json(out / 'config.json', config)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        # Preserve failed runs and re-raise; never fill missing predictions or retry silently.
        config.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(out / 'config.json', config)
        write_json(out / 'progress.json', {'status': 'failed', 'error': config['error']})
        raise


def report(args):
    out = new_output(args.output)
    for script in ['scripts/summarize_paper_comparisons.py', 'scripts/summarize_paper_horizon_generative.py',
                   'scripts/summarize_paper_twostep.py', 'scripts/summarize_paper_probabilistic.py',
                   'scripts/summarize_paper_mouselab_generative.py']:
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)
    sources = {
        'human_prediction': 'runs/paper_matched_comparison_20261003/README.md',
        'horizon_closed_loop': 'runs/paper_horizon_generative_comparison_20261004/README.md',
        'twostep_closed_loop': 'runs/paper_twostep_comparison_20261008/README.md',
        'mouselab_complete_comparison': 'runs/paper_mouselab_model_comparison_20261007/README.md',
        'mouselab_new_model_closed_loop': 'runs/paper_mouselab_generative_comparison_20261008/README.md',
        'probabilistic_event_reports': 'runs/paper_probabilistic_comparison_20261007/README.md',
        'remaining_protocols': 'docs/PAPER_MATCHED_EVALUATION_20261003.md',
        'task_cards': 'docs/PAPER_TASK_BENCHMARK.md',
        'current_model_scope': 'docs/MODEL_SCOPE_20261008.md',
    }
    baseline_status = {}
    baseline_runs = {
        'mouselab_official_policy_simulation': 'runs/paper_mouselab_generative_official_20261008',
        'hybrid_official_linear_ucb_cv': 'runs/paper_hybrid_linear_ucb_cv_20261008',
        'hybrid_official_symbolic_cv': 'runs/paper_hybrid_symbolic_cv_20261008',
    }
    recovery = ROOT / 'runs/paper_mouselab_generative_official_20261008_export'
    if recovery.exists():
        baseline_runs['mouselab_original_attempt'] = baseline_runs['mouselab_official_policy_simulation']
        baseline_runs['mouselab_official_policy_simulation'] = recovery.relative_to(ROOT).as_posix()
    for name, directory in baseline_runs.items():
        run = ROOT / directory
        if not run.exists():
            continue
        summary = run / 'summary.json'
        record_path = summary if summary.exists() else run / 'status.json'
        record = json.loads(record_path.read_text())
        baseline_status[name] = {'status': record['status'],
                                 'complete': record['status'] == 'complete',
                                 'source': record_path.relative_to(ROOT).as_posix()}
        readme = run / 'README.md'
        sources[name] = (readme if readme.exists() else record_path).relative_to(ROOT).as_posix()
    write_json(out / 'report.json', {'status': 'available_results_with_explicit_pending', 'benchmark_complete': False,
                                    'sources': sources, 'supported_prediction_tasks': TASKS,
                                    'supported_generative_tasks': GENERATIVE_TASKS,
                                    'supported_posterior_tasks': POSTERIOR_TASKS,
                                    'official_baseline_runs': baseline_status,
                                    'current_primary_models': ['Laya-English', 'Minitaur8B-current-revision', 'Llama3.1-8B-Instruct'],
                                    'unstarted70B_closed_loop_jobs': 'deferred_by_user; existing completed70B human predictions retained'})
    lines = ['# Public cognitive task evaluation resources', '',
             'Two complete human-prediction tasks accept new models through the shared export and scoring interface.', '',
             'Horizon, Two-step and Mouselab closed-loop runs keep the original environments, and the full-run status of each '
             'model is listed separately. Probabilistic inference keeps the original visible priors and evidence and the event '
             'posterior. Each task supports external model callbacks. Closed-loop reward, human prediction and generated '
             'posterior reports are in separate tables.', '']
    for name, path in sources.items():
        lines.append(f'- [{name}]({(ROOT / path).as_posix()})')
    if baseline_status:
        lines += ['', 'Original CPU baseline runs are listed separately, because an existing task entry does not show that a run is complete.', '']
        lines += [f"- {name}: {record['status']}" for name, record in baseline_status.items()]
    lines += ['', 'The full task matrix still has unfinished cells. Reusing original tasks and implementing interfaces does not by itself '
              'establish the novelty of an independent benchmark.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'available_completed_results', 'output': str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    export = sub.add_parser('export')
    export.add_argument('--task', choices=[*TASKS, *POSTERIOR_TASKS], required=True)
    export.add_argument('--output', type=Path, required=True)
    score = sub.add_parser('score')
    score.add_argument('--task', choices=[*TASKS, *POSTERIOR_TASKS], required=True)
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--task', choices=TASKS, required=True)
    predict.add_argument('--adapter-file', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    predict.add_argument('--device', default='cpu')
    run = sub.add_parser('run')
    run.add_argument('--task', choices=[*GENERATIVE_TASKS, *POSTERIOR_TASKS], required=True)
    run.add_argument('--adapter-file', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--action-rule', choices=['native', 'sample'], default='native')
    run.add_argument('--device', default='cpu')
    summary = sub.add_parser('report')
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps({**TASKS, **GENERATIVE_TASKS, **POSTERIOR_TASKS}, indent=2, ensure_ascii=False))
    elif args.command == 'export':
        export_task(args)
    elif args.command == 'score':
        score_task(args)
    elif args.command == 'predict':
        predict_task(args)
    elif args.command == 'run':
        run_task(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
