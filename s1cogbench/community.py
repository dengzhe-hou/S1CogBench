"""Package completed runs and build a community board without inference or uploads."""
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from s1cogbench import ROOT
from s1cogbench.modes import BY_KEY
from s1cogbench.planning import comparison_context, validate_action_rules

BENCHMARK = 'arxiv-2611.00911'
CONTEXT = ('task', 'protocol', 'mode', 'metric', 'metric_variant', 'scope',
           'target', 'unit', 'aggregation')
METADATA = ('model_name', 'checkpoint', 'revision', 'adapter', 'prompt', 'readout',
            'precision', 'seed', 'hardware', 'command', 'code_revision', 'evidence_url')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def context_key(row):
    return tuple(row[k] for k in CONTEXT)


def numeric(row):
    value, status = row.get('value'), row.get('value_status')
    if status == 'undefined' and value is None:
        return None
    if status == 'defined_positive_infinity' and value == 'Infinity':
        return math.inf
    if status == 'finite' and type(value) in (int, float) and math.isfinite(value):
        return value
    raise ValueError('value and value_status must agree; retain Infinity and undefined values.')


def validate_submission(item):
    if item.get('format_version') != 1 or item.get('benchmark') != BENCHMARK:
        raise ValueError(f'Use submission format 1 and benchmark {BENCHMARK}.')
    sid = item.get('submission_id', '')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', sid):
        raise ValueError('submission_id must use lowercase letters, digits, underscores or hyphens.')
    metadata = item.get('metadata', {})
    for field in METADATA:
        text = metadata.get(field)
        if not isinstance(text, str) or not text.strip() or text.startswith('REPLACE'):
            raise ValueError(f'Fill metadata.{field} with the actual run setting.')
    url = urlsplit(metadata['evidence_url'])
    if url.scheme != 'https' or not url.netloc:
        raise ValueError('evidence_url must be an HTTPS link to the run instructions and evidence.')
    run = item.get('run', {})
    rule = run.get('action_rule')
    if rule not in ('native', 'sample') or type(run.get('probability_readout')) is not bool:
        raise ValueError('run needs action_rule and a boolean probability_readout.')
    rows, coverage = item.get('rows'), item.get('coverage')
    if not isinstance(rows, list) or not rows or not isinstance(coverage, list):
        raise ValueError('Provide nonempty rows and the matching coverage records.')
    if len(rows) != len(coverage):
        raise ValueError('Each result row needs one coverage record.')
    seen = set()
    for row, record in zip(rows, coverage):
        key = record.get('mode_key')
        if key not in BY_KEY or key in seen:
            raise ValueError(f'Unknown or repeated mode_key {key!r}.')
        seen.add(key)
        mode = BY_KEY[key]
        validate_action_rules([mode], rule)
        if record.get('status') != 'complete' or record.get('inputs') != mode.inputs_expected or record.get('scored') != mode.scored_expected:
            raise ValueError(f'{key} needs its complete original input and scored counts.')
        expected = {'task': mode.task, **comparison_context(mode, rule)}
        for field in CONTEXT:
            if row.get(field) != expected[field]:
                raise ValueError(f'{key} has a different {field}; do not relabel a different protocol.')
        if row.get('model') != metadata['model_name'] or row.get('status') != 'user_supplied_full_protocol':
            raise ValueError(f'{key} must identify this model and a complete user-supplied run.')
        if type(row.get('n')) is not int or row['n'] < 0 or row['n'] != record.get('primary_n'):
            raise ValueError(f'{key} needs its original primary-score denominator, not the input count.')
        if not isinstance(row.get('value_raw'), str) or not isinstance(row.get('source'), dict):
            raise ValueError(f'{key} needs value_raw and its source record.')
        numeric(row)
    return item


def prepare(run_dir, metadata_path, submission_id):
    """Read the saved CLI outputs, retaining only aggregates and coverage for publication."""
    run_dir = Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    summary = read_json(run_dir / 'summary.json')
    rows = read_json(run_dir / 'results.json')['rows']
    if config.get('smoke') or summary['complete'] != summary['modes']:
        raise ValueError('Package a completed evaluation, not a smoke run or an incomplete run.')
    modes = config['modes']
    if len(modes) != len(rows) or len(modes) != summary['modes']:
        raise ValueError('Run configuration, summary and result row counts disagree.')
    coverage = []
    for key, row in zip(modes, rows):
        mode = BY_KEY[key]
        record = read_json(run_dir / mode.task / mode.mode / 'mode_result.json')
        primary = record['primary']
        saved_value = primary['value']
        if saved_value == math.inf:
            saved_value = 'Infinity'
        if (row['value'], row['value_status']) != (saved_value, primary['value_status']):
            # Text-only adapters deliberately have no probability-derived NLL.
            absent_probability = (config['probability_readout'] is False and
                                  row['metric'] in ('human_nll', 'aggregate_distribution_nll') and
                                  row['value'] is None and row['value_status'] == 'undefined')
            if not absent_probability:
                raise ValueError(f'{key} aggregate disagrees with its saved mode result.')
        coverage.append({k: record[k] for k in ('status', 'inputs', 'scored')})
        coverage[-1].update(mode_key=key, primary_n=primary['n'])
    item = {'format_version': 1, 'benchmark': BENCHMARK, 'submission_id': submission_id,
            'metadata': read_json(metadata_path),
            'run': {k: config[k] for k in ('s1cogbench_version', 'action_rule', 'probability_readout')},
            'coverage': coverage, 'rows': rows}
    return validate_submission(item)


def load_submissions(directory):
    items = []
    for path in sorted(Path(directory).glob('*.json')):
        item = validate_submission(read_json(path))
        if item['submission_id'] != path.stem:
            raise ValueError(f'{path.name} must match its submission_id.')
        items.append(item)
    return items


def rank_entries(entries, direction):
    """Rank only within an existing target, keeping ties and unscored entries explicit."""
    def sort_key(entry):
        value = numeric(entry)
        score = -value if direction == 'higher' and value is not None else value
        return (value is None, score if direction else 0, entry['model'].lower())

    ordered = sorted(entries, key=sort_key)
    previous, rank = object(), None
    for position, entry in enumerate(ordered, 1):
        value = numeric(entry)
        if direction and value is not None:
            if value != previous:
                rank = position
            previous = value
            entry['rank'] = rank
        else:
            entry['rank'] = None
    return ordered


def build_board(root=ROOT):
    root = Path(root)
    submissions = load_submissions(root / 'submissions')
    aggregate = read_json(root / 'results/aggregate/results.json')
    table = read_json(root / 'results/aggregate/table2.json')
    originals = {r['row_id']: r for r in aggregate['rows']}
    boards = {}
    directions = {}
    for target in table['rows']:
        mode = BY_KEY[target['mode_key']]
        directions[mode.key] = target['direction']
        for cell in target['cells'].values():
            row = originals.get(cell.get('row_id'))
            if row is None:
                continue
            key = context_key(row)
            board = boards.setdefault(key, {'mode_key': mode.key, 'label': mode.label,
                                          'context': {k: row[k] for k in CONTEXT},
                                          'direction': target['direction'], 'entries': []})
            if any(e['row_id'] == row['row_id'] for e in board['entries']):
                continue
            board['entries'].append({**row, 'model_label': aggregate['model_labels'].get(row['model'], row['model']),
                                     'origin': 'paper', 'evidence_url': 'https://arxiv.org/abs/2611.00911'})
    for item in submissions:
        for row, coverage in zip(item['rows'], item['coverage']):
            key = context_key(row)
            mode = BY_KEY[coverage['mode_key']]
            board = boards.setdefault(key, {'mode_key': mode.key, 'label': mode.label,
                                          'context': {k: row[k] for k in CONTEXT},
                                          'direction': directions[mode.key], 'entries': []})
            board['entries'].append({**row, 'model_label': item['metadata']['model_name'],
                                     'origin': 'community', 'submission_id': item['submission_id'],
                                     'metadata': item['metadata'],
                                     'evidence_url': item['metadata']['evidence_url']})
    for board in boards.values():
        board['entries'] = rank_entries(board['entries'], board['direction'])
    return {'benchmark': BENCHMARK, 'submission_count': len(submissions),
            'review_policy': 'Community entries appear after maintainer review and merge. Format checks do not certify scores.',
            'boards': list(boards.values())}
