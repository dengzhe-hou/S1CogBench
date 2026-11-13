"""Extension tasks, contributed paradigms kept apart from the 24-task paper suite.

Each extension task is one file, extensions/<task_id>/task.py, with a TASK
dictionary and either load_trials(root) for the built-in metrics or
run(agent, out, limit, device, action_rule) for its own runner and scorer.
docs/ADD_TASK.md describes both. `--tasks all` keeps selecting only the 27
paper modes. `--tasks extensions` selects every extension task, and an
extension task ID selects that task.
"""
import importlib.util
import math
import os
from pathlib import Path
import re

from s1cogbench import ROOT
from s1cogbench.modes import BY_KEY, MODES, TASKS, Mode, context, select as select_core

DIRECTORY = Path(os.environ.get('S1COGBENCH_EXTENSIONS', ROOT / 'extensions'))
# Built-in metrics: adapter method, better direction and the trial field that holds the target.
METRICS = {'human_nll': ('choice_probabilities', 'lower', 'observed'),
           'accuracy': ('choice_probabilities', 'higher', 'correct'),
           'probability_mae': ('posterior_probability', 'lower', 'target')}
TEXT = ('label', 'paradigm', 'source', 'license', 'split', 'obtain', 'adaptations', 'protocol', 'mode',
        'metric', 'metric_variant', 'unit', 'aggregation', 'target')
REFERENCE = ('value', 'description', 'provenance')


def _load(path):
    spec = importlib.util.spec_from_file_location(f's1cogbench_extension_{path.parent.name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mode(path, module):
    """Validate TASK of one extension file and return its mode."""
    task = getattr(module, 'TASK', None)
    where = f'{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}'
    if not isinstance(task, dict):
        raise ValueError(f'{where} must define a TASK dictionary (see docs/ADD_TASK.md).')
    tid = task.get('id')
    if tid != path.parent.name or not re.fullmatch(r'[a-z][a-z0-9_]*', tid or ''):
        raise ValueError(f'{where}: TASK id must equal the directory name and use lowercase letters, '
                         'digits and underscores.')
    if tid in TASKS or tid in BY_KEY:
        raise ValueError(f'{where}: {tid} is a paper task. Extension tasks need their own ID.')
    for field in TEXT:
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f'{where}: TASK {field} must be a nonempty string.')
    reference = task.get('reference')
    if not isinstance(reference, dict) or any(reference.get(k) in (None, '') for k in REFERENCE):
        raise ValueError(f'{where}: TASK reference needs value, description and provenance.')
    for field in ('n_inputs', 'n_scored'):
        if type(task.get(field)) is not int or task[field] < 1:
            raise ValueError(f'{where}: TASK {field} must be the positive full count.')
    inputs = task.get('inputs')
    if not isinstance(inputs, (list, tuple)) or not all(isinstance(p, str) and p for p in inputs):
        raise ValueError(f'{where}: TASK inputs must list the input paths.')
    custom = callable(getattr(module, 'run', None))
    autonomous = task.get('autonomous', False) is True
    if task['metric'] in METRICS and not custom:
        method, direction, _ = METRICS[task['metric']]
        if not callable(getattr(module, 'load_trials', None)):
            raise ValueError(f'{where}: a built-in metric needs load_trials(root).')
        if autonomous:
            raise ValueError(f'{where}: an autonomous task needs its own run() function.')
    elif custom:
        method, direction = task.get('interface'), task.get('direction')
        if method not in ('choice_probabilities', 'posterior_probability', 'confidence_probability'):
            raise ValueError(f'{where}: run() tasks name the adapter method in TASK interface.')
        if direction not in ('lower', 'higher', 'none'):
            raise ValueError(f'{where}: run() tasks set TASK direction to lower, higher or none.')
    else:
        raise ValueError(f'{where}: metric {task["metric"]!r} is not built in '
                         f'({", ".join(METRICS)}), so the task needs its own run() function.')
    primary = context(task['protocol'] + ('_{rule}' if autonomous else ''), task['mode'], task['metric'],
                      task['metric_variant'], task['unit'], task['aggregation'], task['target'],
                      scope=task.get('scope', 'all'))
    return Mode(tid, 'primary_protocol', 'extension_autonomous' if autonomous else 'extension',
                task['label'], (method,), tuple(inputs), task['n_inputs'], task['n_scored'], primary,
                note='Extension task, not part of the 24-task paper suite.', track='extension',
                direction='' if direction == 'none' else direction)


def discover(directory=DIRECTORY):
    """Modes and modules of extensions/<id>/task.py. Directories starting with _ are templates."""
    modes, modules = [], {}
    for path in sorted(Path(directory).glob('*/task.py')):
        if path.parent.name.startswith(('_', '.')):
            continue
        module = _load(path)
        mode = _mode(path, module)
        modes.append(mode)
        modules[mode.task] = module
    return modes, modules


EXTENSION_MODES, MODULES = discover()
EXTENSION_BY_KEY = {m.key: m for m in EXTENSION_MODES}
ALL_BY_KEY = {**BY_KEY, **EXTENSION_BY_KEY}


def select(names):
    """'all' is the 27 paper modes. 'extensions' adds every extension task; IDs select one task."""
    if names in (['all'], 'all'):
        return list(MODES)
    chosen = []
    for name in names:
        if name == 'extensions':
            hits = list(EXTENSION_MODES)
        elif name in EXTENSION_BY_KEY:
            hits = [EXTENSION_BY_KEY[name]]
        else:
            hits = select_core([name])
        chosen += [m for m in hits if m not in chosen]
    return chosen


# ---------------------------------------------------------------- built-in runner
def _choice(agent, trial):
    keys = list(trial['output_keys'])
    probabilities = [float(p) for p in agent.choice_probabilities(trial['state'], trial['question'], keys)]
    if len(probabilities) != len(keys) or not all(math.isfinite(p) and p >= 0 for p in probabilities) \
            or abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError(f'trial {trial["id"]}: choice_probabilities must return one finite probability '
                         f'per legal key, summing to 1; got {probabilities}')
    native = getattr(agent, 'last_native_choice', None)
    if native not in keys:  # argmax, ties to the first key, as in the paper's autonomous modes
        native = keys[max(range(len(keys)), key=lambda i: (probabilities[i], -i))]
    return {'probabilities': probabilities, 'choice': native}


def score_trials(metric, agent, trials, stream):
    """Score trials with a built-in metric, writing one JSON line per response."""
    import json
    field = METRICS[metric][2]
    total, infinite, scored, invalid = 0.0, False, 0, 0
    for trial in trials:
        row = {'id': trial['id']}
        if metric == 'probability_mae':
            report = agent.posterior_probability(trial['state'], trial['question'])
            row['reported_probability'] = None if report is None else float(report)
            if report is None:  # invalid report: kept in the record, excluded from the mean
                invalid += 1
            else:
                row['absolute_error'] = abs(float(report) - float(trial[field]))
                total += row['absolute_error']
                scored += 1
        else:
            keys = list(trial['output_keys'])
            if trial[field] not in keys:
                raise ValueError(f'trial {trial["id"]}: {field} {trial[field]!r} is not a legal key.')
            row.update(_choice(agent, trial))
            if metric == 'human_nll':
                p = row['probabilities'][keys.index(trial[field])]
                row['nll'] = None if p == 0 else -math.log(p)  # None marks an infinite NLL
                infinite |= p == 0
                total += 0 if p == 0 else row['nll']
            else:
                row['correct'] = row['choice'] == trial[field]
                total += row['correct']
            scored += 1
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    value = None if not scored else total / scored
    return {'status': 'complete', 'metric': metric, 'responses': len(trials), 'scored_responses': scored,
            'invalid_responses': invalid, 'primary_value': None if infinite else value,
            'primary_value_is_infinite': infinite, 'primary_n': scored}
