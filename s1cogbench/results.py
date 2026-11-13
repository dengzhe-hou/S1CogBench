"""Primary-target values from original summaries, written as result-schema rows.

Rows follow results/aggregate/result_schema.json. Only matching protocol,
mode, metric, variant, scope, unit, aggregation and target define a comparison.
Sampled execution remains distinct from native execution.
"""
import csv
import json
import math
from pathlib import Path
import statistics

from s1cogbench.modes import PROBABILITY_METRICS
from s1cogbench.planning import effective_action_rule, comparison_context

COLUMNS = ['row_id', 'task', 'model', 'protocol', 'mode', 'metric', 'metric_variant', 'scope',
           'target', 'unit', 'aggregation', 'n', 'n_raw', 'steps', 'value_raw', 'value_status',
           'status', 'common_metric', 'notes', 'source', 'runtime_context']


def value(v, n, field, **extra):
    if v is None:
        status = 'undefined'
    elif v == math.inf:
        status = 'defined_positive_infinity'
    else:
        status, v = 'finite', float(v)
    return {'value': v, 'value_status': status, 'n': n, 'source_field': field, **extra}


def nll(v, infinite):
    return math.inf if infinite else v


def primary_value(mode, s, out=None):
    """Table 3 primary target of one mode from its original summary `s`."""
    task = mode.task
    if mode.track == 'extension':
        return value(nll(s['primary_value'], s['primary_value_is_infinite']), s['primary_n'],
                     '/primary_value', invalid=s.get('invalid_responses', 0))
    if task == 'bayesian_urn_inference':
        if 'full_code_default400' not in s:  # smoke subset
            return value(s['posterior_mae'], s['scored_rows'], '/posterior_mae')
        return value(s['full_code_default400']['posterior_mae_full_unclipped_additional'],
                     s['full_code_default400']['n_queries'],
                     '/full_code_default400/posterior_mae_full_unclipped_additional')
    if task in ('multiple_cue_judgment', 'multi_attribute_choice'):
        return value(nll(s['typed_choice_mean_nll'], s['typed_choice_mean_nll_is_infinite']),
                     s['responses'], '/typed_choice_mean_nll')
    if task == 'shj_category_learning':
        return value(s['category_accuracy_under_human_history'], s['responses'],
                     '/category_accuracy_under_human_history')
    if task in ('medin_category_learning', 'weather_prediction', 'rlwm', 'recent_probes'):
        return value(s['objective_label_accuracy_under_human_history'], s['objective_scored_responses'],
                     '/objective_label_accuracy_under_human_history')
    if task == 'false_belief_story_qa':
        return value(s['original_answer_exact_match_mean'], s['responses'],
                     '/original_answer_exact_match_mean', invalid=s['invalid_responses'])
    if task in ('politeness_pragmatics', 'pragmatic_language_interpretation'):
        return value(nll(s['human_response_weighted_mean_nll'], s['human_response_mean_nll_is_infinite']),
                     s['responses'], '/human_response_weighted_mean_nll', invalid=s['invalid_responses'])
    if task == 'nonliteral_number_interpretation':
        # Mean over valid reports; all-invalid leaves the MAE undefined.
        return value(s.get('absolute_error_to_human_mean_mean'), s['responses'] - s['invalid_responses'],
                     '/absolute_error_to_human_mean_mean', invalid=s['invalid_responses'])
    if task == 'probability_coherence_judgment':
        g = s['groups']['all']
        return value(g['mean_numeric_absolute_error'], g['numeric_rows'],
                     '/groups/all/mean_numeric_absolute_error')
    if task == 'performance_self_assessment':
        g = s['groups']['all']
        return value(nll(g['mean_typed_nll'], g['mean_typed_nll_is_infinite']), g['responses'],
                     '/groups/all/mean_typed_nll')
    if mode.family == 'prediction':
        return value(nll(s['mean_nll'], s['nll_status'] == 'positive_infinity'), s['decisions'], '/mean_nll')
    if task == 'wcst':
        field = 'human_nll_legal' if mode.mode == 'human_predictive' else 'accuracy'
        parts = s['per_participant']
        return value(statistics.mean(p[field] for p in parts), len(parts),
                     f'/per_participant/*/{field} (participant mean)')
    if task == 'hybrid_information_sampling':
        infinite = s['zero_observed_probability_count'] > 0
        return value(nll(s['pooled_action_nll'], infinite), s['heldout_actions'], '/pooled_action_nll')
    if task == 'nonstationary_bandit':
        return value(s['accuracy_post_step_original'], s['trials'], '/accuracy_post_step_original')
    if task == 'two_step':
        return value(s['mean_reward'], s['n_trials'], '/mean_reward')
    if mode.family == 'horizon_generative':
        return value(s['mean_optimal_choice_rate'], s['n_free_choices'], '/mean_optimal_choice_rate')
    if mode.family == 'mouselab_generative':
        return value(s['mean_terminal_expected_reward'], s['simulations'], '/mean_terminal_expected_reward')
    if task == 'stationary_multiarmed_bandit':
        return value(s['optimal_arm_rate'], s['steps'], '/optimal_arm_rate')
    if task == 'instrumental_learning':
        return value(s['original_latent_mean_optimal_action_rate'], s['data_rows'],
                     '/original_latent_mean_optimal_action_rate')
    if task == 'bart':
        return value(s['mean_reward'], s['data_rows'], '/mean_reward')
    if task == 'temporal_discounting':
        return value(s['delayed_reward_choice_rate'], s['reward_choice_questions'],
                     '/delayed_reward_choice_rate')
    raise KeyError(task)


def encode(v):
    return 'Infinity' if v == math.inf else v


def row(mode, result, model, probabilities=True):
    """One result-schema row for the mode's primary target."""
    rule = result.get('action_rule', 'native')
    c = comparison_context(mode, rule)
    effective_rule = effective_action_rule(mode, rule)
    p = result.get('primary') or {}
    notes = []
    status = 'user_supplied_full_protocol' if result['status'] == 'complete' else 'user_supplied_incomplete'
    v, value_status = p.get('value'), p.get('value_status', 'undefined')
    if result['status'] != 'complete':
        v, value_status = None, 'undefined'
        notes.append('Mode did not complete: ' + result.get('error', 'unknown error'))
    elif not probabilities and c['metric'] in PROBABILITY_METRICS:
        v, value_status = None, 'undefined'
        notes.append('Generated-text adapter: no action probabilities, so this probability '
                     'metric is not applicable (n.a. in Table 3).')
    if p.get('invalid'):
        notes.append(f"{p['invalid']} invalid responses retained in the denominator rules of the original scorer.")
    n = p.get('n')
    return {'row_id': f'{model}/{mode.key}/{c["metric_variant"]}', 'task': mode.task, 'model': model, **c,
            'n': n, 'n_raw': '' if n is None else str(n),
            'value': encode(v), 'value_raw': '' if v is None else ('Infinity' if v == math.inf else repr(v)),
            'value_status': value_status, 'status': status, 'common_metric': True,
            'notes': ' '.join(notes),
            'source': {'asset': f'{mode.task}/{mode.mode}/{result.get("summary_file", "mode_result.json")}',
                       'source_field': p.get('source_field')},
            'runtime_context': {'device': result.get('device'), 'action_rule': rule,
                                'effective_action_rule': effective_rule,
                                'adapter': Path(result.get('adapter', '')).name}}


def write(out, rows):
    out = Path(out)
    (out / 'results.json').write_text(json.dumps(
        {'schema': 'results/aggregate/result_schema.json', 'row_count': len(rows), 'rows': rows},
        ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'steps': '', 'source': json.dumps(r['source'], sort_keys=True),
                             'runtime_context': json.dumps(r['runtime_context'], sort_keys=True)})
