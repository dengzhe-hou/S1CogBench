#!/usr/bin/env python3
"""Rebuild Table 3 from the aggregate files: python scripts/build_table2.py

Reads results/aggregate/results.json and reference_results.json only, picks the
row behind every cell, prints it at the table's precision, and writes
results/aggregate/table2.csv (Table 3 and its Opus and Sonnet columns) and table2.json (each cell
with its file, row_id, value_raw and n). No model call. A selection that matches
no row, or rows with different values, stops the build.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1cogbench.modes import MODES  # noqa: E402

AGG = ROOT / 'results/aggregate'
OPEN = {'Laya': 'laya', 'Intern': 'intern_decision_4b', 'Llama': 'llama8b', 'Minitaur': 'minitaur8b',
        'Qwen': 'qwen35_4b'}
API = {'Jev': 'jev_1_14', 'Opus': 'claude_opus_5_5', 'Sonnet': 'claude_sonnet_5_5', 'Haiku': 'claude_haiku_5_5'}
COLUMNS = ['Uniform', *OPEN, *API, 'Reference']
PRED = 'heldout_human_history_prediction'
ENV = 'autonomous_generated_environment'
SOCIAL = 'aggregate_human_distribution_comparison'
OBJECTIVE = 'objective_label_accuracy_under_human_history'
THREE = ('Laya', 'Llama', 'Minitaur')  # these ran the sampled or original-rule protocol of an autonomous mode


def same(mode, metric, variant, scope='all'):
    """One selection for all five open models."""
    return {c: dict(mode=mode, metric=metric, metric_variant=variant, scope=scope) for c in OPEN}


def split(three, two):
    """Autonomous modes: Laya, Llama and Minitaur under one row family, Intern and Qwen under another."""
    return {c: (three if c in THREE else two) for c in OPEN}


NLL = {'mode': PRED, 'metric': 'human_nll', 'scope': 'all'}
# Domain, Table 3 row, primary target, direction, percent, decimals, open-model selection,
# reference row_id, reference mark, reference type. The rows follow Table 3's order.
TABLE = [
    ('Evidence', 'Bayesian urn', 'Posterior MAE', 'lower', False, 3,
     same('generated_explicit_posterior', 'mae', 'posterior_mae'),
     'bayesian_urn_analytic-posterior_mae', '', 'optimal'),
    ('Evidence', 'Multiple-cue judgment', 'Human NLL', 'lower', False, 3,
     {c: dict(NLL, metric_variant='typed_choice_mean_nll', protocol=p) for c, p in {
         'Laya': 'compact_full_history', 'Intern': 'compact_full_history', 'Llama': 'original_full_prefix_typed',
         'Minitaur': 'psych101-extended-full-prefix-typed-v1', 'Qwen': 'compact_full_history'}.items()},
     'mcpl-overall-test-nll', '', 'human fit'),
    ('Evidence', 'Multi-attribute choice', 'Human NLL', 'lower', False, 3,
     same(PRED, 'human_nll', 'typed_choice_mean_nll'),
     'multi_attribute_wadd-typed_choice_mean_nll', '†', 'human fit'),
    ('Evidence', 'Probability coherence', 'Human MAE (pp)', 'lower', False, 2,
     same(PRED, 'mae', 'mean_numeric_absolute_error'),
     'probability_coherence_reference_20261030-mean_numeric_absolute_error', '¶', 'human'),
    ('Category', 'SHJ categories', 'Accuracy (%)', 'higher', True, 1,
     same(PRED, 'accuracy', 'category_accuracy_under_human_history'),
     'shj_gcm-category_accuracy_under_human_history', '†', 'human fit'),
    ('Category', 'Medin categories', 'Accuracy (%)', 'higher', True, 1,
     same(PRED, 'accuracy', OBJECTIVE), f'medin_gcm-{OBJECTIVE}', '†', 'human fit'),
    ('Category', 'Weather prediction', 'Accuracy (%)', 'higher', True, 1,
     same(PRED, 'accuracy', OBJECTIVE), f'weather_gcm-{OBJECTIVE}', '†', 'human fit'),
    ('History', 'RLWM', 'Accuracy (%)', 'higher', True, 1,
     same('original_human_history_replay_no_official_heldout_split', 'accuracy', OBJECTIVE),
     'rlwm-M9-argmax-accuracy', '§', 'human fit'),
    ('History', 'Recent probes', 'Accuracy (%)', 'higher', True, 1,
     same(PRED, 'accuracy', OBJECTIVE), f'recent_probes_rational-{OBJECTIVE}', '†', 'human fit'),
    ('Adaptation', 'WCST, prediction', 'Human NLL', 'lower', False, 3,
     same('predictive', 'human_nll', 'participant_mean', 'predictive'), 'wcst-sl-2-human_nll', '', 'human fit'),
    ('Adaptation', 'WCST, autonomous', 'Accuracy (%)', 'higher', True, 1,
     same('generative', 'accuracy', 'participant_mean', 'generative'), 'wcst-sl-3-objective_accuracy', '', 'human fit'),
    ('Adaptation', 'Restless bandit', 'Better arm (%)', 'higher', True, 1,
     same(ENV, 'accuracy', 'accuracy_post_step_original'),
     'restless_qlearning-accuracy_post_step_original', '∗', 'algorithm'),
    ('Adaptation', 'Instrumental learning', 'Better arm (%)', 'higher', True, 1,
     same(ENV, 'accuracy', 'original_latent_mean_optimal_action_rate'),
     'instrumental_learning_reference_20261030-original_latent_mean_optimal_action_rate', '¶', 'algorithm'),
    ('Exploration', 'Horizon, prediction', 'Human NLL', 'lower', False, 3,
     same(PRED, 'human_nll', 'legal_nll_eps1e8'), 'rw3-heldout-legal-nll', '', 'human fit'),
    ('Exploration', 'Horizon, autonomous', 'Optimal (%)', 'higher', True, 1,
     split(dict(mode=ENV, metric='accuracy', metric_variant='original_latent_mean_optimal_action_rate', scope='all'),
           dict(mode='autonomous_generative', metric='accuracy', metric_variant='optimal_action_rate', scope='free_choices')),
     'rw3-autonomous-optimal_action_rate', '', 'human fit'),
    ('Exploration', 'Stationary bandits', 'Optimal arm (%)', 'higher', True, 1,
     split(dict(mode='autonomous_summary_history', metric='accuracy', metric_variant='original_latent_mean_optimal_action_rate', scope='all'),
           dict(mode='autonomous', metric='accuracy', metric_variant='accuracy', scope='all_16_original_conditions')),
     'stationary_thompson_family-accuracy', '†', 'algorithm'),
    ('Information', 'Hybrid sampling', 'Human NLL', 'lower', False, 3,
     same(PRED, 'human_nll', 'pooled_action_nll'), 'hybrid-ann-human_nll', '‡', 'human fit'),
    ('Information', 'Mouselab, prediction', 'Human NLL', 'lower', False, 3,
     {c: dict(NLL, metric_variant='strict_sdk_mean_nll' if c == 'Laya' else 'mean_nll') for c in OPEN},
     'mouselab-metagreedy-2', '‡', 'human fit'),
    ('Information', 'Mouselab, autonomous', 'Expected return', 'higher', False, 2,
     split(dict(mode=ENV, metric='reward', metric_variant='mean_terminal_expected_reward', scope='all'),
           dict(mode='autonomous_generative', metric='reward', metric_variant='mean_terminal_expected_reward', scope='native_autonomous_full')),
     'mouselab_metagreedy_autonomous-mean_terminal_expected_reward', '∗', 'human fit'),
    ('Preference', 'BART', 'Mean reward', 'higher', False, 2,
     same(ENV, 'reward', 'mean_reward'), 'bart_reference_20261030-mean_reward', '¶', 'algorithm'),
    ('Preference', 'Temporal discounting', 'Delayed reward (%)', '', True, 1,
     same(ENV, 'preference', 'delayed_reward_choice_rate', 'reward_choice_questions'),
     'temporal_discounting_reference_20261030-delayed_reward_choice_rate', '¶', 'human'),
    ('Planning', 'Two-step', 'Mean reward', 'higher', False, 2,
     same(ENV, 'reward', 'mean_reward'), 'two_step_reference_20261030-mean_reward', '¶', 'human fit'),
    ('Social', 'False-belief QA', 'Accuracy (%)', 'higher', True, 1,
     same('public_scenario_qa', 'accuracy', 'original_answer_exact_match_mean'),
     'false_belief_answer_key-original_answer_exact_match_mean', '', 'optimal'),
    ('Social', 'Politeness', 'Aggregate NLL', 'lower', False, 3,
     same(SOCIAL, 'aggregate_distribution_nll', 'human_response_weighted_mean_nll'),
     'politeness_reference_20261030-human_response_weighted_mean_nll', '¶', 'human'),
    ('Social', 'Pragmatic interpretation', 'Aggregate NLL', 'lower', False, 3,
     same(SOCIAL, 'aggregate_distribution_nll', 'human_response_weighted_mean_nll'),
     'pragmatic_interpretation_reference_20261030-human_response_weighted_mean_nll', '¶', 'human'),
    ('Social', 'Nonliteral number', 'MAE to human mean', 'lower', False, 2,
     {**same(SOCIAL, 'mae', 'absolute_error_to_human_mean_mean'),
      'Minitaur': dict(mode='original_source_scenarios', metric='mae', metric_variant='mae', scope='all')},
     'nonliteral_number_reference_20261030-absolute_error_to_human_mean_mean', '¶', 'human'),
    ('Metacognition', 'Self-assessment', 'Human NLL', 'lower', False, 3,
     same(PRED, 'human_nll', 'mean_typed_nll'),
     'self_assessment_lookup_reference_20261030-i_within_logic_10fold_crossfit', '¶', 'human fit'),
]
MODE_OF = {m.label: m for m in MODES}


def printed(row, percent, decimals):
    """The cell as Table 3 prints it. An undefined value without a marker is the all-invalid MAE."""
    if row.get('table2_marker'):
        return row['table2_marker']
    if row['value_status'] == 'undefined':
        return 'inv.'
    if row['value'] == 'Infinity':
        text = '+∞'
    else:
        text = f'{100 * row["value"]:.{decimals}f}' if percent else f'{row["value"]:.{decimals}f}'
    return text + ('s' if (row.get('runtime_context') or {}).get('readout') == 'stated probability' else '')


def pick(rows, task, model, selection):
    hits = [r for r in rows if r['task'] == task and r['model'] == model
            and all(r.get(k) == v for k, v in selection.items())]
    values = {r['value_raw'] for r in hits}
    if len(values) != 1:
        raise SystemExit(f'{task}/{model}/{selection}: {len(hits)} rows, values {sorted(values)}')
    return hits[0]


def cell(row, name, percent, decimals):
    return {'printed': printed(row, percent, decimals), 'file': name, 'row_id': row['row_id'],
            'value_raw': row['value_raw'], 'n': row['n']}


def score(cell):
    raw = cell['value_raw']
    return float('inf') if raw == 'Infinity' else float(raw) if raw else None


def mark_open_cells(cells, direction):
    """Table 3's bold and gray, among the five open models only. No mark where no direction is better."""
    if not direction:
        return
    sign = 1 if direction == 'lower' else -1
    scored = {c: sign * score(cells[c]) for c in OPEN if score(cells[c]) is not None}
    best = min(scored.values())
    uniform = sign * score(cells['Uniform'])
    for c, value in scored.items():
        cells[c]['best_open'] = value == best
        cells[c]['not_better_than_uniform'] = value >= uniform


def main():
    rows = json.loads((AGG / 'results.json').read_text())['rows']
    refs = {r['row_id']: r for r in json.loads((AGG / 'reference_results.json').read_text())['rows']}
    api = {(r['mode_key'], r['model']): r for r in rows
           if r.get('mode_key') and r['model'] in API.values() and not r['metric_variant'].endswith('clipped_1e12')}
    table = []
    for domain, label, target, direction, percent, decimals, selection, reference, mark, kind in TABLE:
        mode = MODE_OF[label]
        cells = {'Uniform': cell(refs[f'uniform-{mode.key.replace(":", "-")}'], 'reference_results.json', percent, decimals)}
        for column, model in OPEN.items():
            cells[column] = cell(pick(rows, mode.task, model, selection[column]), 'results.json', percent, decimals)
        for column, model in API.items():
            cells[column] = cell(api[(mode.key, model)], 'results.json', percent, decimals)
        cells['Reference'] = cell(refs[reference], 'reference_results.json', percent, decimals)
        mark_open_cells(cells, direction)
        table.append({'domain': domain, 'table2_row': label, 'mode_key': mode.key, 'primary_target': target,
                      'direction': direction, 'reference_mark': mark, 'reference_type': kind, 'cells': cells})
    (AGG / 'table2.json').write_text(json.dumps({
        'description': 'Table 3 with the row behind every cell. Written by scripts/build_table2.py from results.json '
                       'and reference_results.json.',
        'columns': COLUMNS, 'rows': table}, indent=1, ensure_ascii=False) + '\n')
    with (AGG / 'table2.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Domain', 'Task mode', 'Primary target', 'Better direction', *COLUMNS, 'Reference mark', 'Reference type'])
        for r in table:
            writer.writerow([r['domain'], r['table2_row'], r['primary_target'], r['direction'],
                             *(r['cells'][c]['printed'] for c in COLUMNS), r['reference_mark'], r['reference_type']])
    print(f'{len(table)} rows and {sum(len(r["cells"]) for r in table)} cells written to {AGG}')


if __name__ == '__main__':
    main()
