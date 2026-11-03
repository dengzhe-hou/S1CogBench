"""Original CogBench urn questions with an explicit event-probability interface.

The task/scoring methods execute from the public source. Fixed replay seeds and
typed event reports are explicit adaptations of the original numeric decoder.
"""
import ast
import contextlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.paper_horizon import load_agent, write_json

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external/CogBench/Experiments/ProbabilisticReasoning'
N_RUNS = 200
PAPER_RUNS = 100
SEED = 20261007
SCORE_COLUMNS = [
    'engine', 'run', 'performance_score1', 'performance_score1_name',
    'performance_score1_CI', 'behaviour_score1', 'behaviour_score1_name',
    'behaviour_score1_CI', 'behaviour_score2', 'behaviour_score2_name',
    'behaviour_score2_CI', 'behaviour_score3', 'behaviour_score3_name',
    'behaviour_score3_CI', 'behaviour_score4', 'behaviour_score4_name',
    'behaviour_score4_CI',
]


def _source_methods(path, class_name, methods, namespace=None):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    if {node.name for node in selected} != set(methods):
        raise ValueError(f'Published methods missing in {path}.')
    scope = {'np': np, 'pd': pd, 'sm': sm, **(namespace or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return type(class_name, (), {name: scope[name] for name in methods})


def load_official(context=None, version='1'):
    """Original methods, with capture of hidden source labels outside requests."""
    if version not in {'1', '2', '3'}:
        raise ValueError('Original version must be 1, 2 or 3.')
    context = context if context is not None else {}
    original_urn = _source_methods(UPSTREAM / 'envs/urns_exp.py', 'UrnTask',
                                   ['__init__', 'step'])

    class CapturingUrn(original_urn):
        def step(self, left_urn, right_urn):
            observation, urn = super().step(left_urn, right_urn)
            context.update(informative_lh=bool(self.informative_lh),
                           prior=float(self.left_prob), lh=float(self.red_prob),
                           red_observation=observation == 'red',
                           left_urn=left_urn, right_urn=right_urn,
                           generated_urn_is_left=urn == left_urn)
            return observation, urn

    envs = SimpleNamespace(urns_exp=SimpleNamespace(UrnTask=CapturingUrn))
    cls = _source_methods(UPSTREAM / 'query.py', 'LearningToInferExpForLLM',
                          ['run_single_experiment', 'random_fct', 'del_letters_at_end'],
                          {'envs': envs})
    task = cls()
    task.parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(version_number=version))
    return task


def event_question(metadata, version='1'):
    """P(true) is source identity, never a choice or correctness confidence."""
    left = metadata['left_urn']
    noun, observation = {
        '1': ('urn', 'ball'), '2': ('dice', 'face'), '3': ('mine', 'gem'),
    }[version]
    return {
        'type': 'noul',
        'instructions': f'The observed {observation} came from {noun} {left}.',
    }


class CaptureLLM:
    Q_A = ('\n\nQ:', '\n\nA:')

    def __init__(self, context, records, version):
        self.context, self.records, self.version = context, records, version

    def generate(self, prompt):
        metadata = {key: self.context[key] for key in [
            'run', 'environment_seed', 'informative_lh', 'prior', 'lh',
            'red_observation', 'left_urn', 'right_urn',
        ]}
        likelihood = metadata['lh'] if metadata['red_observation'] else 1 - metadata['lh']
        prior = metadata['prior']
        posterior = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))
        condition = 'likelihood' if metadata['informative_lh'] else 'prior'
        self.records.append({
            'request': {
                'id': f'{metadata["run"]}/{condition}', 'state': prompt,
                'question': event_question(metadata, self.version),
                'original_llm_default_prompt': prompt + self.Q_A[1] + ' ' + self.format_answer,
            },
            'target': {
                'bayes_posterior': posterior,
                'generated_urn_is_left': self.context['generated_urn_is_left'],
            },
            'metadata': metadata,
        })
        # Capture the original task prompt without invoking a neural model or
        # consuming random numbers through the original invalid-answer fallback.
        return '50'


def load_records(seed=SEED, version='1'):
    """All 200 current-code runs; first 100 are the declared paper-scale subset."""
    context, records = {}, []
    task = load_official(context, version)
    llm = CaptureLLM(context, records, version)
    previous_random_state = np.random.get_state()
    try:
        for run in range(N_RUNS):
            context.update(run=run, environment_seed=seed + run)
            np.random.seed(seed + run)
            frame = task.run_single_experiment(llm)
            if len(frame) != 2 or frame.informative_lh.tolist() != [True, False]:
                raise ValueError('Original run must contain both information conditions.')
    finally:
        np.random.set_state(previous_random_state)
    if len(records) != N_RUNS * 2 or len({row['request']['id'] for row in records}) != N_RUNS * 2:
        raise ValueError('Incomplete original code-default question set.')
    return records


def load_official_scorer():
    cls = _source_methods(UPSTREAM / 'store.py', 'StoringPriorLhScores', ['get_scores'])
    return cls()


def prediction_map(predictions):
    """Explicit numeric reports, with complete ID coverage checked by scoring."""
    if isinstance(predictions, dict):
        return {key: float(value['reported_event_probability'] if isinstance(value, dict) else value)
                for key, value in predictions.items()}
    result = {}
    for prediction in predictions:
        key = prediction['id']
        if key in result:
            raise ValueError(f'Duplicate prediction id: {key}')
        result[key] = float(prediction['reported_event_probability'])
    return result


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def score_rows(records, predictions):
    supplied = prediction_map(predictions)
    expected = {row['request']['id'] for row in records}
    if len(expected) != len(records) or set(supplied) != expected:
        raise ValueError('Predictions must cover each question exactly once, with no extra IDs.')
    scored = []
    for row in records:
        key = row['request']['id']
        probability = supplied[key]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f'{key}: return a finite event probability in [0,1].')
        target = row['target']
        label = bool(target['generated_urn_is_left'])
        label_probability = probability if label else 1 - probability
        scored.append({
            'id': key, **row['metadata'],
            'left_pred': probability, 'reported_event_probability': probability,
            'reported_event_probability_round2': round(probability, 2),
            'bayes_posterior': float(target['bayes_posterior']),
            'generated_urn_is_left': label,
            'posterior_absolute_error': abs(probability - target['bayes_posterior']),
            'event_brier': (probability - int(label)) ** 2,
            'event_nll': -math.log(label_probability) if label_probability > 0 else None,
        })
    return scored


def _score_subset(rows, paper_intercept=False):
    frame = pd.DataFrame(rows)
    likelihood = np.where(frame.red_observation, frame.lh, 1 - frame.lh)
    posterior = frame.prior * likelihood / (
        frame.prior * likelihood + (1 - frame.prior) * (1 - likelihood))
    clipped = frame.left_pred.clip(1e-5, 1 - 1e-5)
    errors = (posterior - clipped).abs()
    threshold = float(errors.quantile(.95))
    kept = errors < threshold
    with contextlib.redirect_stdout(io.StringIO()):
        table = load_official_scorer().get_scores(
            frame.copy(deep=True), pd.DataFrame(columns=SCORE_COLUMNS),
            'external_event_probability', run=0)
    official = table.iloc[0]
    infinite = sum(row['event_nll'] is None for row in rows)
    result = {
        'n_queries': len(rows), 'n_runs': int(frame.run.nunique()),
        'posterior_mae_full_unclipped_additional': float(frame.posterior_absolute_error.mean()),
        'posterior_mae_full_original_clipping': float(errors.mean()),
        'official_performance_score1': _finite(official.performance_score1),
        'official_performance_status': 'undefined_no_strictly_below_q95_errors'
            if not kept.any() else 'finite',
        'official_performance_ci_halfwidth': _finite(official.performance_score1_CI),
        'official_trim_threshold_q95': threshold, 'official_trim_kept_count': int(kept.sum()),
        'prior_weight_original_no_intercept_OLS': _finite(official.behaviour_score1),
        'prior_weight_ci_halfwidth_original_OLS': _finite(official.behaviour_score1_CI),
        'likelihood_weight_original_no_intercept_OLS': _finite(official.behaviour_score2),
        'likelihood_weight_ci_halfwidth_original_OLS': _finite(official.behaviour_score2_CI),
        'prior_likelihood_ratio_original': _finite(official.behaviour_score3),
        'prior_minus_likelihood_original': _finite(official.behaviour_score4),
        'event_brier_additional': float(frame.event_brier.mean()),
        'event_nll_additional': None if infinite else math.fsum(row['event_nll'] for row in rows) / len(rows),
        'event_nll_status': 'positive_infinity' if infinite else 'finite',
        'zero_actual_event_probability_count': infinite,
    }
    if paper_intercept:
        inputs = sm.add_constant(np.column_stack([
            np.log(frame.prior / (1 - frame.prior)), np.log(likelihood / (1 - likelihood)),
        ]), has_constant='add')
        fit = sm.OLS(np.log(clipped / (1 - clipped)), inputs).fit()
        result['paper_formula_with_intercept_additional'] = {
            'intercept': float(fit.params.iloc[0]), 'prior_weight': float(fit.params.iloc[1]),
            'likelihood_weight': float(fit.params.iloc[2]),
        }
    return result


def score_records(records, predictions, paper_intercept=True):
    """Original code score plus full MAE; never silently repair undefined scores."""
    rows = score_rows(records, predictions)
    if len(rows) != 400 or {row['run'] for row in rows} != set(range(N_RUNS)):
        raise ValueError('Score the complete code-default 200 runs / 400 questions.')
    paper_rows = [row for row in rows if row['run'] < PAPER_RUNS]
    if len(paper_rows) != 200:
        raise ValueError('Incomplete predeclared first-100-run paper-scale subset.')
    rounded = [{**row, 'left_pred': row['reported_event_probability_round2'],
                'posterior_absolute_error': abs(row['reported_event_probability_round2'] - row['bayes_posterior']),
                'event_brier': (row['reported_event_probability_round2'] - int(row['generated_urn_is_left'])) ** 2,
                'event_nll': (-math.log(row['reported_event_probability_round2'] if row['generated_urn_is_left']
                    else 1 - row['reported_event_probability_round2']))
                    if (row['reported_event_probability_round2'] if row['generated_urn_is_left']
                        else 1 - row['reported_event_probability_round2']) > 0 else None,
               } for row in rows]
    bayes_rows = score_rows(records, {row['request']['id']: row['target']['bayes_posterior']
                                      for row in records})
    return {
        'status': 'complete', 'task': 'probabilistic-posterior', 'n_runs': N_RUNS,
        'n_queries': len(rows),
        'data_kind': 'program_generated_environment_not_human_data',
        'score_status': 'current_official_code_scoring_with_new_typed_event_probability_interface',
        'original_scoring': 'Clip q to [1e-5,1-1e-5]; OLS logit(q) on prior and '
                            'observed-likelihood log odds without intercept; performance '
                            '1-minus-MAE strictly below error q95. No undefined-score repair.',
        'full_code_default400': _score_subset(rows, paper_intercept),
        'paper_scale_first100runs200': _score_subset(paper_rows, paper_intercept),
        'by_information_condition': {
            name: _score_subset([row for row in rows if row['informative_lh'] == flag], paper_intercept)
            for name, flag in [('informative_likelihood', True), ('informative_prior', False)]
        },
        'reported_round2_additional': _score_subset(rounded, paper_intercept),
        'exact_bayes_reference': _score_subset(bayes_rows, paper_intercept),
        'event_metrics_status': 'Additional event Brier/NLL use the true generated source urn, '
                                'captured only as a scoring label. No human labels or reward implied.',
        'paper_scale_status': 'Predeclared nested first100 runs, retaining both original conditions; '
                              'paper states100 simulations whereas current query defaults200 runs.',
        'probability_precision': 'Submitted q preserved; separate round(q,2) additional comparison. '
                                 'This does not reproduce the original numeric-token decoder.',
    }


def run_probabilistic(output, adapter_file, device='cpu'):
    """Run all current-code v1 questions using a frozen external P(event) model."""
    output, adapter_file = Path(output), Path(adapter_file)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    config = {
        'status': 'loading_agent', 'task': 'probabilistic-posterior', 'version_number': '1',
        'adapter_file': str(adapter_file.resolve()), 'device': device,
        'n_runs': N_RUNS, 'expected_queries': N_RUNS * 2, 'paper_scale_runs': PAPER_RUNS,
        'seed': SEED, 'environment_seeds': {str(run): SEED + run for run in range(N_RUNS)},
        'source': str(UPSTREAM / 'query.py'), 'scorer': str(UPSTREAM / 'store.py'),
        'data_kind': 'new_environment_samples_from_original_public_generator_not_human_data',
        'seeding_status': 'New per-run replay seeds; all models receive identical saved questions.',
        'model_input': 'Only original visible prompt state and explicit source-identity event question. '
                       'No hidden source label, exact posterior, reward or future observations.',
        'interface': 'create_agent(device) returns posterior_probability(state,question) -> q in [0,1]. '
                     'New explicit event-probability interface, not original two-token numeric decoder.',
        'model_configuration': 'Frozen; no tuning, context override or precision change by runner.',
        'timing_context': 'Shared machine; timings are not exclusive-device performance.',
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'statsmodels']},
    }
    write_json(output / 'config.json', config)
    predictions = []
    agent = None
    current_request = None
    try:
        records = load_records()
        with (output / 'requests.jsonl').open('w') as requests, (output / 'targets.jsonl').open('w') as targets:
            for row in records:
                requests.write(json.dumps(row['request'], ensure_ascii=False, allow_nan=False) + '\n')
                targets.write(json.dumps({'id': row['request']['id'], **row['target'],
                                         'metadata': row['metadata']}, allow_nan=False) + '\n')
        load_started = time.monotonic()
        agent = load_agent(adapter_file, device)
        config.update(status='running', load_seconds=time.monotonic() - load_started,
                      agent_type=f'{type(agent).__module__}.{type(agent).__qualname__}')
        if hasattr(agent, 'get_agent_metadata'):
            config['agent_metadata'] = agent.get_agent_metadata()
        elif hasattr(agent, 'metadata'):
            config['agent_metadata'] = agent.metadata
        write_json(output / 'config.json', config)
        with (output / 'predictions.jsonl').open('w') as file:
            for row in records:
                request = row['request']
                current_request = request
                begin = time.monotonic()
                value = float(agent.posterior_probability(request['state'], request['question']))
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{request["id"]}: return a finite event probability in [0,1].')
                prediction = {'id': request['id'], 'reported_event_probability': value,
                              'inference_seconds': time.monotonic() - begin}
                for source, target in [('last_input_tokens', 'input_tokens'),
                                       ('last_raw_required_tokens', 'raw_required_tokens'),
                                       ('last_prompt', 'actual_adapter_prompt'),
                                       ('last_raw_answer', 'raw_adapter_answer')]:
                    if hasattr(agent, source):
                        prediction[target] = getattr(agent, source)
                predictions.append(prediction)
                file.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
                if len(predictions) % 20 == 0:
                    file.flush()
                    progress = {'status': 'running', 'completed_queries': len(predictions),
                                'expected_queries': 400, 'elapsed_seconds': time.monotonic() - started}
                    write_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        rows = score_rows(records, predictions)
        pd.DataFrame(rows).to_csv(output / 'trials.csv', index=False)
        summary = score_records(records, predictions)
        summary.update(seed=SEED, version_number='1', adapter_file=config['adapter_file'],
                       load_seconds=config['load_seconds'], total_seconds=time.monotonic() - started,
                       total_inference_seconds=math.fsum(row['inference_seconds'] for row in predictions),
                       timing_context=config['timing_context'])
        tokens = [row['input_tokens'] for row in predictions if 'input_tokens' in row]
        if tokens:
            summary.update(min_input_tokens=min(tokens), max_input_tokens=max(tokens))
        write_json(output / 'summary.json', summary)
        write_json(output / 'progress.json', summary)
        config['status'] = 'complete'
        write_json(output / 'config.json', config)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    except Exception as error:
        failure = {'status': 'failed', 'task': config['task'], 'completed_queries': len(predictions),
                   'expected_queries': 400, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'total_seconds': time.monotonic() - started}
        if current_request is not None:
            failure['failed_request_id'] = current_request['id']
        if agent is not None:
            for source, target in [('last_raw_answer', 'raw_adapter_answer'),
                                   ('last_prompt', 'actual_adapter_prompt')]:
                if hasattr(agent, source):
                    failure[target] = getattr(agent, source)
        config.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'config.json', config)
        write_json(output / 'failure.json', failure)
        write_json(output / 'progress.json', failure)
        raise
