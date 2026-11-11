#!/usr/bin/env python3
"""Build the seven-model release from accepted aggregate inputs, without inference."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_table2 import cell, score
from s1cogbench.results import COLUMNS
from s1cogbench.modes import BY_KEY

ARCHIVE = ROOT / 'results/aggregate'
LATEST = ROOT / 'results/latest'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def build():
    sources = load(LATEST / 'primary_sources.json')
    controls = load(LATEST / 'readout_sources.json')['rows']
    aggregate = load(ARCHIVE / 'results.json')
    table = load(ARCHIVE / 'table2.json')
    originals = {row['row_id']: row for row in aggregate['rows']}
    targets = {row['mode_key']: row for row in table['rows']}
    models = sources['models']
    primary_ids = []
    for point_index, point in enumerate(sources['primary_points']):
        target = targets[point['mode_key']]
        label, model = point['model_label'], point['model']
        previous = target['cells'].get(label)
        if previous:
            row = originals[previous['row_id']]
        else:
            row = {'row_id': f"latest/{model}/{point['mode_key']}", 'model': model, 'model_access': 'open',
                   **point['context'], 'status': 'available_full_protocol', 'common_metric': True}
            aggregate['rows'].append(row)
        row.update(value=point['value'], value_raw=str(point['value']), n=point['n'],
                   n_raw=str(point['n']), value_status='defined_positive_infinity' if point['value'] == 'Infinity' else 'finite',
                   release_id=sources['release_id'], mode_key=point['mode_key'],
                   runtime_context=point['runtime_context'],
                   source={'asset': 'primary_sources.json',
                           'source_field': f"/primary_points/{point_index}",
                           'retained_source': point['source'], 'original_scorer': point['scorer']})
        primary_ids.append(row['row_id'])
        precision = len(target['cells']['Uniform']['printed'].split('.')[-1])
        percent = '%' in target['primary_target']
        selected = cell(row, 'results.json', percent, precision)
        assert selected['printed'] == point['expected_printed'], (point['mode_key'], label, selected)
        target['cells'][label] = selected
    for target in table['rows']:
        cells = target['cells']
        if target['direction']:
            sign = 1 if target['direction'] == 'lower' else -1
            values = {name: sign * score(cells[name]) for name in models}
            ranked = sorted(set(values.values()))
            uniform = sign * score(cells['Uniform'])
            for name, value in values.items():
                cells[name].update(best_open=value == ranked[0], second_open=len(ranked) > 1 and value == ranked[1],
                                   not_better_than_uniform=value >= uniform - 1e-12)
        target['cells'] = {name: cells[name] for name in ['Uniform', *models, 'Jev', 'Opus', 'Sonnet', 'Haiku', 'Reference']}
    # Shared legal readouts retain separate protocols from the main native-readout table.
    for point in controls:
        task, metric, rule = point['task'], point['metric'], point['readout']
        mode = BY_KEY[f'{task}:autonomous_generative']
        is_accuracy = metric in ('accuracy', 'optimal_choice_rate')
        row = {'row_id': f"readout/{point['model']}/{task}/{rule}/{metric}", 'model': point['model'],
               'task': task, 'mode_key': mode.key, 'protocol': f'p3_original_source_legal_{rule}',
               'mode': 'autonomous_generative' if task == 'horizon' else 'generative',
               'metric': 'accuracy' if is_accuracy else 'reward' if metric == 'mean_reward_free' else metric,
               'metric_variant': ('optimal_action_rate' if task == 'horizon' else 'participant_mean') if is_accuracy else metric,
               'scope': 'free_choices' if task == 'horizon' else 'generative',
               'target': mode.primary['target'] if is_accuracy else 'source_' + metric,
               'unit': 'task_specific_reward_units' if metric == 'mean_reward_free' else 'fraction',
               'aggregation': 'response_weighted_mean' if task == 'horizon' else 'participant_mean',
               'n': point['decisions'] if task == 'horizon' else point['participants'],
               'value': point['value'], 'value_raw': repr(point['value']), 'value_status': 'finite',
               'status': 'available_full_protocol', 'common_metric': True,
               'release_id': sources['release_id'],
               'source': {'asset': 'readout_sources.json', 'retained_source': point['source'],
                          'source_acceptance': point['source_acceptance']},
               'runtime_context': {'effective_action_rule': rule,
                                   'readout': 'legal-option argmax' if rule == 'argmax' else 'categorical legal-option sampling'},
               'notes': 'Original model-specific prompts, task seeds and own action-dependent histories retained. '
                        'Nox main native/argmax used A100 MIG; added sampling used a whole A100. '
                        'CLM retains the first 2048 tokens. These are performance scores, not matched-hardware timings.',
               'leaderboard_direction': 'higher' if is_accuracy or metric == 'mean_reward_free' else 'lower',
               'leaderboard_label': mode.label + ' · shared legal ' + rule}
        aggregate['rows'].append(row)
    aggregate['model_labels'].update(nox4b='Decision-2.0-Nox-4B', clm8b='CLM-v0.1-8B')
    aggregate['historical_aggregate_snapshot_utc'] = aggregate.pop('snapshot_utc')
    aggregate.update(release_id=sources['release_id'], snapshot_date='2026-11-10',
                     primary_model_count=7, primary_row_ids=primary_ids,
                     scope='Seven-model primary release and accepted shared-readout controls. Historical additional aggregate rows remain source-linked.',
                     row_count=len(aggregate['rows']))
    table.update(description='Latest seven-model primary scores, checked against the accepted manuscript table. The original arXiv table remains under results/aggregate/.',
                 release_id=sources['release_id'], columns=['Uniform', *models, 'Jev', 'Opus', 'Sonnet', 'Haiku', 'Reference'])
    save(LATEST / 'results.json', aggregate)
    save(LATEST / 'table2.json', table)
    with (LATEST / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in aggregate['rows']:
            writer.writerow({**row, 'source': json.dumps(row.get('source', {})),
                             'runtime_context': json.dumps(row.get('runtime_context', {}))})
    with (LATEST / 'table2.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Domain', 'Task mode', 'Primary target', 'Better direction', *table['columns']])
        for row in table['rows']:
            writer.writerow([row['domain'], row['table2_row'], row['primary_target'], row['direction'],
                             *(row['cells'][name]['printed'] for name in table['columns'])])
    print(f"Built {len(primary_ids)} primary scores, {len(controls)} readout metrics, and 27 table rows.")


if __name__ == '__main__':
    build()
