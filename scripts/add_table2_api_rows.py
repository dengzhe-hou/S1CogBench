#!/usr/bin/env python3
"""Add Table 3's API columns and new Reference values to results/aggregate.

    python scripts/add_table2_api_rows.py VALUES.json

VALUES.json holds source_acceptance, accepted_utc, models, cells and references,
exported from the closed acceptance records of the Table 3 cells. Existing rows
stay byte for byte: JSON files are rewritten with their original serialization
and the CSV is only appended to. A new row takes the comparison context of its
mode's Table 3 primary target (s1cogbench/modes.py), so it groups with the
accepted open-model rows; a cell's own protocol replaces the primary one where
the run used another history. A marker cell (n.a., part., n.e., n.f., canc.,
inv.) becomes a row with no value whose status names the marker. A cell's n is
the aggregation denominator; its model_calls is the number of requests.
"""
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1cogbench.modes import BY_KEY  # noqa: E402

AGG = ROOT / 'results/aggregate'
MARKERS = {'n.a.': 'not_applicable_readout', 'part.': 'partial_records_retained', 'n.e.': 'not_evaluated',
           'n.f.': 'non_full_prefix_only', 'canc.': 'cancelled_before_generation', 'inv.': 'all_responses_invalid'}
READOUTS = {'native typed choice', 'stated probability', 'generated FINAL line'}
# Variant labels of the values file mapped to the scorer field they average.
VARIANTS = {'source epsilon 1e-12': 'typed_choice_mean_nll_clipped_1e12'}
BY_LABEL = {m.label: m.key for m in BY_KEY.values()}
COLUMNS = ['row_id', 'task', 'model', 'protocol', 'mode', 'metric', 'metric_variant', 'scope', 'target', 'unit',
           'aggregation', 'n', 'n_raw', 'steps', 'value_raw', 'value_status', 'status', 'common_metric', 'notes',
           'source', 'runtime_context']


def load(name):
    return json.loads((AGG / name).read_text())


def dump(name, value):
    (AGG / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_value(raw, status):
    if status == 'finite':
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f'finite value expected, got {raw!r}')
        return value
    if status == 'defined_positive_infinity':
        return 'Infinity'
    if status == 'undefined':
        return None
    raise ValueError(f'unknown value_status {status!r}')


def mode_key(entry):
    """A cell names its mode by key (python -m s1cogbench list) or by its Table 3 row label."""
    key = entry.get('mode') or BY_LABEL.get(entry.get('table2_row'))
    if key not in BY_KEY:
        raise ValueError(f'Unknown mode {entry.get("mode")!r} / Table 3 row {entry.get("table2_row")!r}')
    if entry.get('table2_row') and BY_LABEL.get(entry['table2_row']) != key:
        raise ValueError(f'Table 3 row {entry["table2_row"]!r} is not mode {key!r}')
    return key


def variant(label):
    if label is None or label in VARIANTS.values() or ' ' not in label:
        return label
    if label not in VARIANTS:
        raise ValueError(f'Unknown metric variant label {label!r}; add its scorer field to VARIANTS.')
    return VARIANTS[label]


def context(key):
    mode = BY_KEY[key]
    c = {k: v.format(rule='native') for k, v in mode.primary.items()}
    if mode.family == 'mouselab_generative':
        c['scope'] = 'native_autonomous_full'
    return mode, c


def cell_row(i, cell, models, source):
    key = mode_key(cell)
    mode, c = context(key)
    model = cell['model']
    if model not in models:
        raise ValueError(f'{key}: model {model!r} is not declared in "models"')
    marker = cell.get('marker')
    if marker is not None and marker not in MARKERS:
        raise ValueError(f'{key}/{model}: unknown marker {marker!r}')
    if marker is None and cell.get('readout') not in READOUTS:
        raise ValueError(f'{key}/{model}: readout must be one of {sorted(READOUTS)}')
    if cell.get('metric_variant'):
        c['metric_variant'] = variant(cell['metric_variant'])
    if cell.get('protocol'):
        c['protocol'] = cell['protocol']
    status = cell.get('value_status', 'undefined' if marker else 'finite')
    value = None if marker else parse_value(cell['value_raw'], status)
    n = cell.get('n')
    notes = [f'Table 3 API column ({models[model]["label"]}).']
    if marker:
        notes.append(f'Table 3 prints {marker}')
    else:
        notes.append(f'Readout: {cell["readout"]}.')
    if cell.get('metric_variant') and variant(cell['metric_variant']) != cell['metric_variant']:
        notes.append(f'Variant labelled "{cell["metric_variant"]}" in Table 3 records.')
    if cell.get('note'):
        notes.append(cell['note'])
    return {'row_id': f'a{i:05d}', 'task': mode.task, 'model': model, **c, 'mode_key': key,
            'value_status': status if not marker else 'undefined',
            'status': MARKERS[marker] if marker else 'available_full_protocol',
            'notes': ' '.join(notes), 'n_raw': '' if n is None else str(n), 'n': n,
            'value_raw': '' if marker else cell['value_raw'], 'value': value, 'common_metric': True,
            'source': {'values_file': source, 'record': cell.get('source_record'), 'field': cell.get('source_field')},
            'runtime_context': {'device': 'provider_hosted', 'api_snapshot': models[model].get('snapshot'),
                                'readout': None if marker else cell['readout'],
                                'model_calls': cell.get('model_calls')},
            'table2_marker': marker}


def reference_row(ref, source):
    mode, c = context(mode_key(ref))
    status = ref.get('value_status', 'finite')
    return {'row_id': f'{ref["reference_id"]}-{variant(ref["metric_variant"])}', 'reference_id': ref['reference_id'],
            'model': ref['label'], 'task': mode.task, 'mode': c['mode'], 'metric': ref['metric'],
            'value': parse_value(ref['value_raw'], status), 'value_raw': ref['value_raw'], 'value_status': status,
            'n': ref['n'], 'n_detail': ref.get('n_detail'), 'aggregation': ref['aggregation'], 'unit': ref['unit'],
            'scope': ref['scope'],
            'provenance_type': ref['provenance_type'], 'fitting_exposure': ref['fitting_exposure'],
            'comparison_note': ref['comparison_note'], 'exact_precision': True,
            'precision': 'saved_aggregate_decimal', 'status': 'accepted_completed_reference',
            'source': {'values_file': source, 'record': ref['source_record'], 'field': ref['source_field']},
            'metric_variant': variant(ref['metric_variant']), 'table2_reference_type': ref.get('ref_type'),
            'table2_mark': ref.get('table2_mark'), 'table2_metric': ref.get('table2_metric')}


def main():
    values = json.loads(Path(sys.argv[1]).read_text())
    source = values['source_acceptance']
    models = values['models']
    results = load('results.json')
    if any(r['model'] in models for r in results['rows']):
        raise SystemExit('API rows are already present; start from the committed aggregate files.')
    rows = [cell_row(i, c, models, source) for i, c in enumerate(values['cells'], 1)]
    seen = Counter((r['mode_key'], r['model'], r['metric_variant']) for r in rows)
    if max(seen.values(), default=0) > 1:
        raise SystemExit(f'Duplicate cells: {[k for k, v in seen.items() if v > 1]}')

    old_rows = list(results['rows'])
    results['rows'] = old_rows + rows
    results['row_count'] = len(results['rows'])
    results['model_labels'].update({k: m['label'] for k, m in models.items()})
    results['model_access'] = {**{k: 'open_weights' for k in results['model_labels'] if k not in models},
                               **{k: 'api' for k in models}}
    results['model_access']['gaussian_ts'] = 'classical_policy'
    results['api_extension'] = {'source_acceptance': source, 'accepted_utc': values.get('accepted_utc'),
                                'rows_added': len(rows), 'models': models,
                                'markers': {m: MARKERS[m] for m in MARKERS}}
    dump('results.json', results)

    with (AGG / 'results.csv').open(newline='') as f:
        raw = f.read()
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\r\n')
    for r in rows:
        writer.writerow(['' if r.get(k) is None else (json.dumps(r[k], sort_keys=True) if k in ('source', 'runtime_context')
                                                      else r[k]) for k in COLUMNS])
    with (AGG / 'results.csv').open('w', newline='') as f:
        f.write(raw + out.getvalue())

    refs = load('reference_results.json')
    new_refs = [reference_row(x, source) for x in values.get('references', [])]
    refs['rows'] += new_refs
    refs['row_count'] = len(refs['rows'])
    refs['table2_extension'] = {'source_acceptance': source, 'rows_added': len(new_refs)}
    dump('reference_results.json', refs)
    provenance = load('baseline_provenance.json')
    known = {b['id'] for b in provenance['baselines']}
    for x in values.get('references', []):
        if x['reference_id'] not in known:
            mode, _ = context(mode_key(x))
            provenance['baselines'].append({
                'id': x['reference_id'], 'label': x['label'], 'task': mode.task, 'status': 'completed_reference',
                'provenance_type': x['provenance_type'], 'fitting_exposure': x['fitting_exposure'],
                'comparison_note': x['comparison_note'], 'table2_reference_type': x.get('ref_type')})
            known.add(x['reference_id'])
    dump('baseline_provenance.json', provenance)

    manifest = load('data_manifest.json')
    counts = manifest['counts']
    counts['public_result_rows'] = len(results['rows'])
    counts['public_models'].update(Counter(r['model'] for r in rows))
    counts['public_value_status_counts'] = dict(Counter(r['value_status'] for r in results['rows']))
    counts['table2_marker_rows'] = dict(Counter(r['table2_marker'] for r in rows if r['table2_marker']))
    manifest['scope']['api_models'] = len(models)
    manifest['scope'].pop('jev_calls', None)
    manifest['accepted_input_snapshots'].append(source)
    manifest['limitations'] = [x for x in manifest['limitations'] if not x.startswith('Jev is deferred')] + [
        'API rows keep the provider readout: native typed choice (Jev), generated FINAL line (Claude, '
        'human-NLL targets n.a.) or stated probabilities (Claude Haiku on four modes). They are not internal '
        'probabilities of open weights.']
    dump('data_manifest.json', manifest)

    tasks = load('tasks.json')
    per_task = Counter(r['task'] for r in rows)
    modes_of = Counter(m.task for m in BY_KEY.values())
    full_modes = {(r['task'], r['mode_key'], r['model']) for r in rows if not r['table2_marker']}
    cells_full = Counter((task, model) for task, _, model in full_modes)
    for t in tasks['tasks']:
        if t['task_id'] in per_task:
            t['public_result_rows'] += per_task[t['task_id']]
            t['full_sample_api_models'] = sorted(m for m in models if cells_full[(t['task_id'], m)] == modes_of[t['task_id']])
    tasks['api_models'] = list(models)
    tasks['api_full_sample_modes'] = {m: sum(1 for _, _, model in full_modes if model == m) for m in models}
    jev = [m for m in models if m.startswith('jev')]
    if jev:
        tasks['jev_status'] = f"api_full_sample_modes_{tasks['api_full_sample_modes'][jev[0]]}_of_27"
    dump('tasks.json', tasks)
    print(json.dumps({'rows_added': len(rows), 'references_added': len(new_refs),
                      'markers': counts['table2_marker_rows'],
                      'api_full_sample_modes': tasks['api_full_sample_modes']}, indent=1))
    with (AGG / 'results.csv').open(newline='') as f:
        assert f.read().startswith(raw) and results['rows'][:len(old_rows)] == old_rows


if __name__ == '__main__':
    main()
