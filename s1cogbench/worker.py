"""Run the modes of one task through their original runners and scorers.

`python -m s1cogbench eval` starts one worker process per task, in the same
order as scripts/validate_cpu_reuse_24.py, so module state and global random
streams start fresh for every task. Each family function below performs the
same calls as the matching function of that validation driver, with the
user's adapter in place of the CPU reference policy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

from s1cogbench import ROOT, smoke
from s1cogbench.modes import BY_KEY
from s1cogbench.results import primary_value

sys.path.insert(0, str(ROOT))
COGBENCH = {'instrumental_learning': 'instrumental', 'bart': 'bart', 'temporal_discounting': 'temporal'}
# Full-sample run counts of the accepted results. BART's original code default
# is 10 runs; the five accepted model rows use 20 (900 balloons).
EXPANSION_RUNS = {'instrumental': 10, 'bart': 20, 'temporal': 1}
EXPANSION_ROWS = {'instrumental': 96, 'bart': 45, 'temporal': 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def module(name):
    value = importlib.import_module(name)
    if not Path(value.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f'{name} was imported from {value.__file__}, not from {ROOT}. '
                           'Remove other S1CogBench copies from PYTHONPATH.')
    return value


def finite(value):
    if value is None:
        return {'value': None, 'value_status': 'undefined'}
    if value == math.inf:
        return {'value': math.inf, 'value_status': 'defined_positive_infinity'}
    return {'value': float(value), 'value_status': 'finite'}


class Context:
    def __init__(self, args):
        self.adapter = Path(args.adapter).resolve()
        self.device, self.action_rule, self.label = args.device, args.action_rule, args.model_label
        self.smoke, self.smoke_calls, self.limit, self.trace = args.smoke, args.smoke_calls, 0, []
        self._agent = None
        # Runners that load the adapter file themselves get the smoke shim,
        # which returns the same adapter instance under the same call budget.
        self.runner_adapter = ROOT / 's1cogbench/smoke_adapter.py' if self.smoke else self.adapter

    def begin(self, mode):
        if self.smoke:
            self.limit, self.trace = smoke.LIMITS.get(mode.task, self.smoke_calls), []
            smoke.configure(self.adapter, self.limit, self.trace)
            smoke.STATE['agent'] = self._agent

    def agent(self):
        if self.smoke:
            budget = smoke.budget_agent(self.device)
            self._agent = smoke.STATE['agent']
            return budget
        if self._agent is None:
            self._agent = module('src.paper_horizon').load_agent(self.adapter, self.device)
        return self._agent

    def check(self, count, expected, what):
        if not self.smoke and count != expected:
            raise ValueError(f'{what}: {count} != expected {expected}')


def unique(ids, expected, ctx):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate original source IDs.')
    ctx.check(len(ids), expected, 'Full unique source IDs')


def event_prediction(agent, trial):
    """Single-slider social items are explicit event-probability reports."""
    started = time.monotonic()
    value = agent.posterior_probability(trial.state, trial.question)
    return {'reported_probability': None if value is None else float(value),
            'batch_elapsed_seconds': time.monotonic() - started,
            'inference_interface': 'adapter_posterior_probability'}


# ---------------------------------------------------------------- families
def typed(mode, out, ctx):
    task = mode.task
    if task == 'shj_category_learning':
        m = module('src.psych_typed')
        trials = m.load_shj_trials()
        inventory, score, summarize = m.source_inventory(trials), m.score_prediction, m.summarize
    elif task in {'medin_category_learning', 'weather_prediction'}:
        m = module('src.psych_category_tasks')
        trials = m.load_category_trials('medin' if task.startswith('medin') else 'weather')
        inventory, score, summarize = m.inventory(trials), m.score, m.summarize_category
    elif task == 'rlwm':
        m = module('src.rlwm_typed')
        original = m.read_source()
        trials = m.build_trials(original)
        inventory, score, summarize = m.inventory(original, trials), m.score, m.summarize
    else:
        name = ('src.psych_compact' if task == 'multiple_cue_judgment' else
                'src.psych_extended' if task in {'multi_attribute_choice', 'recent_probes'}
                else 'src.psych_social_tasks')
        m = module(name)
        trials = m.load_trials(task)
        inventory, score, summarize = m.inventory(trials), m.score, m.summarize
    expected = mode.inputs_expected
    if task == 'medin_category_learning':
        assert inventory['response_kinds'] == {
            'learning/category_choice': 6300, 'test/category_choice': 336,
            'test/typicality_rating': 336}, inventory['response_kinds']
    if task == 'rlwm':
        assert inventory['source_rows'] == 53573, inventory['source_rows']
    unique([t.id for t in trials], expected, ctx)
    save(out / 'source_inventory.json', inventory)
    if ctx.smoke:
        trials, expected = trials[:ctx.limit], min(ctx.limit, len(trials))
    agent = ctx.agent()
    predict = module('scripts.run_psych_typed').predict_original_scalar
    rows = []
    with (out / 'responses.jsonl').open('w') as stream:
        for i, t in enumerate(trials):
            if getattr(t, 'kind', None) == 'single-slider':
                prediction = event_prediction(agent, t)
            else:
                prediction = predict(agent, t, ctx.device)
            row = score(t, prediction)
            row['runner_batch_id'] = i
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            rows.append(row)
    result = summarize(rows, expected)
    assert result['status'] == 'complete'
    save(out / 'original_summary.json', result)
    return {'inputs': len(trials), 'scored': len(rows), 'summary_file': 'original_summary.json',
            'summary': result}


def four(mode, out, ctx):
    m = module('scripts.run_four_task_responses_20261018')
    argument = ('probability_coherence_judgment' if mode.task == 'probability_coherence_judgment'
                else 'answer_self_assessment')
    ids, kinds = [], Counter()
    for trials, metadata in m.sessions(argument):
        for t in trials:
            ids.append(t.id)
            kinds[f'{t.experiment}:{t.response_kind}'] += 1
    unique(ids, mode.inputs_expected, ctx)
    if ctx.smoke:
        return four_smoke(m, argument, out, ctx)
    dest = out / 'original_runner'
    m.run(SimpleNamespace(task=argument, output=dest, adapter_file=ctx.runner_adapter,
                          model_label=ctx.label, device=ctx.device))
    s = json.loads((dest / 'summary.json').read_text())
    assert s['status'] == 'complete' and s['full_scope_complete']
    return {'inputs': len(ids), 'scored': s['responses'], 'summary_file': 'original_runner/summary.json',
            'summary': s, 'kinds': dict(kinds)}


def four_smoke(m, argument, out, ctx):
    """The runner's per-response path (predict_original_scalar, score_response) on the first items."""
    agent, rows = ctx.agent(), []
    for trials, metadata in m.sessions(argument):
        for trial in trials:
            if len(rows) == ctx.limit:
                break
            rows.append(m.score_response(trial, m.predict_original_scalar(agent, trial, ctx.device)))
    with (out / 'responses.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    errors = [r['native_scale_absolute_error'] for r in rows if r.get('native_scale_absolute_error') is not None]
    nll = [r['typed_choice_nll'] for r in rows]
    summary = {'groups': {'all': {
        'responses': len(rows), 'numeric_rows': len(errors),
        'mean_numeric_absolute_error': sum(errors) / len(errors) if errors else None,
        'mean_typed_nll_is_infinite': any(x is None for x in nll),
        'mean_typed_nll': None if any(x is None for x in nll) else sum(nll) / len(nll)}}}
    save(out / 'original_summary.json', summary)
    return {'inputs': len(rows), 'scored': len(rows), 'summary_file': 'original_summary.json',
            'summary': summary}


def posterior(mode, out, ctx):
    m = module('src.paper_probabilistic')
    records = m.load_records()
    unique([x['request']['id'] for x in records], 400, ctx)
    if ctx.smoke:
        records = records[:ctx.limit]
    agent, predictions = ctx.agent(), []
    with (out / 'predictions.jsonl').open('w') as stream:
        for row in records:
            q = row['request']
            value = agent.posterior_probability(q['state'], q['question'])
            if value is None:
                raise ValueError(f"No event probability for {q['id']}: the Bayesian-urn scorer "
                                 'has no invalid-response rule, so every query needs a report.')
            prediction = {'id': q['id'], 'reported_event_probability': float(value)}
            stream.write(json.dumps(prediction, allow_nan=False) + '\n')
            predictions.append(prediction)
    scored = m.score_rows(records, predictions)
    with (out / 'scored_rows.jsonl').open('w') as stream:
        for row in scored:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    if ctx.smoke:
        errors = [r['posterior_absolute_error'] for r in scored]
        summary = {'scored_rows': len(scored), 'posterior_mae': sum(errors) / len(errors)}
    else:
        summary = m.score_records(records, predictions)
    save(out / 'original_summary.json', summary)
    return {'inputs': len(records), 'scored': len(scored), 'summary_file': 'original_summary.json',
            'summary': summary}


def prediction(mode, out, ctx):
    m = module('scripts.benchmark_paper_tasks')
    arg = 'horizon-predictive' if mode.task == 'horizon' else 'mouselab-predictive'
    records = m.load_records(arg)
    unique([x['request']['id'] for x in records], mode.inputs_expected, ctx)
    if ctx.smoke:
        agent, rows = ctx.agent(), []
        for record in records[:ctx.limit]:
            keys = record['request']['output_keys']
            p = agent.choice_probabilities(record['request']['state'], record['request']['question'], keys)
            rows.append(m.probability_scores(record, dict(zip(keys, map(float, p))),
                                             m.TASKS[arg]['epsilon']))
        summary = m.aggregate(rows)
        save(out / 'original_summary.json', summary)
        return {'inputs': len(rows), 'scored': len(rows), 'summary_file': 'original_summary.json',
                'summary': summary}
    del records
    dest = out / 'original_runner'
    m.predict_task(SimpleNamespace(task=arg, output=dest, adapter_file=ctx.runner_adapter,
                                   device=ctx.device))
    s = json.loads((dest / 'summary.json').read_text())
    return {'inputs': mode.inputs_expected, 'scored': s['decisions'],
            'summary_file': 'original_runner/summary.json', 'summary': s}


def wcst(mode, out, ctx):
    m = module('src.wcst_task')
    scorer = module('scripts.run_wcst_models')
    data = m.test_data()
    ctx.check(sum(map(len, data.values())), 12754, 'WCST test trials')
    a, parts = ctx.agent(), []
    original_mode = 'predictive' if mode.mode == 'human_predictive' else 'generative'
    if ctx.smoke:
        data = dict(list(data.items())[:1])
        data = {sid: rows[:ctx.limit] for sid, rows in data.items()}
    with (out / 'decisions.jsonl').open('w') as stream:
        for sid, source in data.items():
            dialogue, rows = m.Dialogue('centaur', original_mode), []
            for i, row in enumerate(source):
                state = dialogue.prompt(row)
                p = a.choice_probabilities(state, m.QUESTION, list(m.LETTERS))
                native = getattr(a, 'last_native_choice', None)
                if native not in m.LETTERS:
                    raise ValueError('WCST runs the original native-choice rule: the adapter must set '
                                     f'last_native_choice to one of {m.LETTERS}, got {native!r}.')
                key = m.LETTERS.index(native) + 1
                assert len(p) == 4 and abs(sum(p) - 1) < 1e-6
                record = {'subject_id': sid, 'trial': i, 'key': key,
                          'human_key': int(row['user_key']), 'ground_key': int(row['ground_key']),
                          'legal_probabilities': [float(x) for x in p], 'native_choice': native}
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                rows.append(record)
                actual = int(row['user_key']) if original_mode == 'predictive' else key
                dialogue.update(row, actual, native if original_mode == 'generative' else None)
            part = {'subject_id': sid, 'mode': original_mode, 'trials': len(rows),
                    **m.task_metrics(source, [x['key'] for x in rows])}
            if original_mode == 'predictive':
                part.update(scorer.predictive_metrics(rows, False))
            parts.append(part)
    count = sum(x['trials'] for x in parts)
    summary = {'status': 'complete', 'mode': original_mode, 'responses': count,
               'participants': len(parts), 'per_participant': parts,
               'original_full_vocabulary_nll': None,
               'original_full_vocabulary_nll_status': 'Not produced by the typed choice interface.'}
    save(out / 'original_summary.json', summary)
    return {'inputs': count, 'scored': count, 'summary_file': 'original_summary.json',
            'summary': summary}


def hybrid(mode, out, ctx):
    m = module('scripts.run_hybrid_open_prediction')
    export = out / 'full_export'
    m.export(export)
    info = json.loads((export / 'summary.json').read_text())
    ctx.check(info['requests'], 163135, 'Hybrid held-out requests')
    dest = out / 'original_runner'
    m.run(dest, export, ctx.runner_adapter, ctx.device, False, 1)
    s = json.loads((dest / 'summary.json').read_text())
    assert s['status'] == 'complete'
    return {'inputs': info['requests'], 'scored': s['heldout_actions'],
            'summary_file': 'original_runner/summary.json', 'summary': s}


def restless(mode, out, ctx):
    dest = out / 'original_runner'
    dest.mkdir()
    s = module('src.paper_restless').run_restless(dest, ctx.runner_adapter, ctx.device)
    ctx.check(s['queries'], 1600, 'Restless queries')
    return {'inputs': s['queries'], 'scored': s['queries'], 'summary_file': 'original_runner/summary.json',
            'summary': s}


def twostep(mode, out, ctx):
    dest = out / 'original_runner'
    dest.mkdir()
    module('src.paper_twostep').run_twostep(dest, ctx.runner_adapter, ctx.device, ctx.action_rule)
    s = json.loads((dest / 'summary.json').read_text())
    ctx.check(s['n_decisions'], 4000, 'Two-step decisions')
    return {'inputs': s['n_decisions'], 'scored': s['n_decisions'],
            'summary_file': 'original_runner/summary.json', 'summary': s}


def horizon_generative(mode, out, ctx):
    dest = out / 'original_runner'
    dest.mkdir()
    module('src.paper_horizon').run_generative(dest, ctx.runner_adapter, ctx.device, ctx.action_rule)
    s = json.loads((dest / 'summary.json').read_text())
    assert s['n_rows'] == 74400 and s['n_participants'] == 31
    ctx.check(s['n_free_choices'], 34720, 'Horizon free choices')
    return {'inputs': s['n_free_choices'], 'scored': s['n_free_choices'],
            'summary_file': 'original_runner/summary.json', 'summary': s}


def mouselab_generative(mode, out, ctx):
    dest = out / 'original_runner'
    dest.mkdir()
    m = module('src.paper_mouselab_generative')
    people = json.loads((ROOT / 'data/mouselab_callaway2022_official/trials.json').read_text()).keys()
    slots = list(m.episode_schedule(people))
    ctx.check(len(slots), 23750, 'Mouselab episode slots')
    required = [ROOT / '.tools/julia-1.4.1/bin/julia', ROOT / '.tools/mouselab-julia141-env/Project.toml',
                ROOT / '.tools/mouselab-julia141-env/Manifest.toml', m.MODEL_COPY, m.SERVER]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Julia runtime or Mouselab model assets missing: ' + repr(missing))
    if ctx.smoke:
        servers, original = [], m.MouselabServer

        class Tracked(original):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                servers.append(self)

        m.MouselabServer = Tracked
        try:
            m.run_mouselab_generative(dest, ctx.runner_adapter, ctx.device, ctx.action_rule)
        finally:
            m.MouselabServer = original
            for server in servers:
                if server.process.poll() is None:
                    server.close(abort=True)
    s = m.run_mouselab_generative(dest, ctx.runner_adapter, ctx.device, ctx.action_rule)
    ctx.check(s['simulations'], 23750, 'Mouselab simulations')
    return {'inputs': len(slots), 'scored': s['simulations'], 'summary_file': 'original_runner/summary.json',
            'summary': s}


def bandit(mode, out, ctx):
    m = module('scripts.run_bandit_open_models')
    bench = m.FullBench(num_threads=1)
    specs = m.specifications(bench)
    expected = sum(x['horizon'] for x in specs) * m.N_TRAJECTORIES
    ctx.check(len(specs), 16, 'Bandit conditions')
    ctx.check(expected, 312000, 'Bandit scheduled steps')
    a, episodes, correct = ctx.agent(), [], 0
    with (out / 'episodes.jsonl').open('w') as stream:
        for spec, env in zip(specs, bench.verbal_envs):
            means = spec['arm_expected_rewards_scoring_only']
            for repeat in range(m.N_TRAJECTORIES):
                row = m.run_episode(env, spec, repeat, a, 'summary', ctx.action_rule, out)
                steps = [json.loads(line) for line in (out / row['trajectory_file']).open()]
                # Optimal-arm rate: the chosen arm has the maximal latent mean (ties count).
                row['optimal_arm_steps'] = sum(means[r['chosen_action_index']] == max(means) for r in steps)
                correct += row['optimal_arm_steps']
                episodes.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
    count = sum(x['steps'] for x in episodes)
    ctx.check(len(episodes), 480, 'Bandit episodes')
    summary = {'status': 'complete', 'conditions': 16, 'episodes': len(episodes), 'steps': count,
               'history': 'summary', 'action_rule': ctx.action_rule,
               'optimal_arm_rate': correct / count,
               'per_episode_original_reward_and_pseudo_regret': episodes,
               'source_protocol': m.SOURCE_LABEL,
               'note': 'Original helper executed; different reward scales are not pooled.'}
    save(out / 'original_summary.json', summary)
    return {'inputs': len(specs), 'scored': count, 'summary_file': 'original_summary.json',
            'summary': summary}


def expansion(mode, out, ctx):
    import pandas as pd
    m = module('src.cogbench_expansion')
    arg = COGBENCH[mode.task]
    runs = EXPANSION_RUNS[arg]
    expected = runs * EXPANSION_ROWS[arg]
    frames = []
    with (out / 'requests.jsonl').open('w') as stream:
        bridge = m.ChoiceLLM(ctx.agent(), arg, stream, ctx.action_rule, ctx.device)
        for i in range(runs):
            frame = m.run_one(arg, bridge, i, 10000 + i, '1', 500)
            frame.to_csv(out / f'run_{i:04d}.csv', index=False)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(out / 'trials.csv', index=False)
        ctx.check(len(frame), expected, f'{arg} original rows')
        scores = m.score_runs(arg, frame, ctx.label, 10000)
        scores.to_csv(out / 'original_scores.csv', index=False)
        summary = m.summary_metrics(arg, frame, bridge.records)
    summary = {'status': 'complete', 'task': arg, 'runs': runs, 'original_scoring': 'complete',
               'original_scores_file': 'original_scores.csv', **summary}
    if arg == 'instrumental':
        chosen = frame.mean0.where(frame.choice == 0, frame.mean1)
        best = frame[['mean0', 'mean1']].max(axis=1)
        summary['original_latent_mean_optimal_action_rate'] = float((chosen == best).mean())
    if arg == 'temporal':
        summary.update(temporal_items(bridge.records))
    save(out / 'original_summary.json', summary)
    return {'inputs': len(frame), 'scored': len(frame), 'summary_file': 'original_summary.json',
            'summary': summary}


def temporal_items(records):
    """Table 3 preference metric (2026-10-29): later-receipt choices among reward
    questions; payment-only questions excluded. Same item logic as the job that
    computed the accepted Table 3 cells."""
    import re
    reward = delayed = 0
    for z in records:
        q = z['question']['criteria']
        assert set(q) == {'1', '2'} and z['valid'] and z['choice'] in q
        if all(x.startswith('Pay ') for x in q.values()):
            continue
        months = {k: (int(re.search(r'(\d+) months', x).group(1)) if re.search(r'(\d+) months', x) else 0)
                  for k, x in q.items()}
        assert len(set(months.values())) == 2 and all(
            ('receive' in x.lower() or 'dollars' in x.lower()) for x in q.values())
        reward += 1
        delayed += z['choice'] == max(months, key=months.get)
    return {'reward_choice_questions': reward, 'delayed_reward_choices': delayed,
            'delayed_reward_choice_rate': delayed / reward if reward else None}


FAMILIES = {'typed': typed, 'four': four, 'posterior': posterior, 'prediction': prediction,
            'wcst': wcst, 'hybrid': hybrid, 'restless': restless, 'twostep': twostep,
            'horizon_generative': horizon_generative, 'mouselab_generative': mouselab_generative,
            'bandit': bandit, 'expansion': expansion}


def run_mode(mode, out, ctx):
    started = time.monotonic()
    ctx.begin(mode)
    result = {'task': mode.task, 'mode': mode.mode, 'key': mode.key, 'family': mode.family,
              'adapter': str(ctx.adapter), 'device': ctx.device, 'action_rule': ctx.action_rule,
              'started_utc': now(), 'smoke_limit': ctx.limit or None}
    try:
        done = FAMILIES[mode.family](mode, out, ctx)
        result.update(status='complete', inputs=done['inputs'], scored=done['scored'],
                      summary_file=done['summary_file'],
                      primary=primary_value(mode, done['summary'], out))
    except smoke.SmokeStop:
        result.update(status='complete', primary=None)
    except Exception as exc:  # noqa: BLE001 - recorded, printed and returned as a failed mode
        hint = (f' This mode calls {", ".join(mode.interface)} on the adapter.'
                if isinstance(exc, AttributeError) else '')
        result.update(status='failed', error=repr(exc) + hint, traceback=traceback.format_exc())
        traceback.print_exc()
    result.update(finished_utc=now(), seconds=time.monotonic() - started,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if ctx.smoke:
        result['smoke_trace'] = list(ctx.trace)
    save(out / 'mode_result.json', result)
    print(json.dumps({k: result.get(k) for k in ('key', 'status', 'inputs', 'scored', 'primary', 'seconds')},
                     allow_nan=True), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes', required=True, help='Comma-separated mode keys of one task.')
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--action-rule', choices=['native', 'sample'], default='native')
    p.add_argument('--model-label', default='user_model')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-calls', type=int, default=smoke.LIMIT)
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(args.threads)
    ctx = Context(args)
    failed = 0
    for key in args.modes.split(','):
        mode = BY_KEY[key]
        out = args.output / mode.task / mode.mode
        out.mkdir(parents=True, exist_ok=False)
        failed += run_mode(mode, out, ctx)['status'] != 'complete'
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
