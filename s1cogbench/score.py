"""Reliability against uniform choice for the lite targets of an `eval` output directory.

For each target, the model's difference from Uniform gets a 95% percentile bootstrap interval over the
target's sampling units (participants, runs, items or story blocks; 10,000 resamples), as in the paper's
interval table. Within a resample the score is recomputed as the scorer does it, sum(num)/sum(den), +inf if any
resampled unit holds an infinite record, and a record-dependent Uniform is recomputed on the same units.
A target counts as reliably better when the whole interval lies on the better side.

The point value is checked against the run's mode_result.json before any interval is reported.
"""
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

B, SEED = 10_000, 20261103
LN = math.log
# Table 3 row index (seed), label, direction, printed in percent, resampling unit, constant Uniform.
TARGETS = {
    'multi_attribute_choice': (2, 'Multi-attribute choice', 'lower', False, 'participant', None),
    'nonstationary_bandit': (11, 'Restless bandit', 'higher', True, 'run', .5),
    'instrumental_learning': (12, 'Instrumental learning', 'higher', True, 'run', None),
    'horizon:human_predictive': (13, 'Horizon, prediction', 'lower', False, 'participant', LN(2)),
    'two_step': (21, 'Two-step', 'higher', False, 'run', None),
    'false_belief_story_qa': (22, 'False-belief QA', 'higher', True, 'story block', None),
    'politeness_pragmatics': (23, 'Politeness', 'lower', False, 'item', None),
    'pragmatic_language_interpretation': (24, 'Pragmatic interpretation', 'lower', False, 'item', None),
}


def truthy(x):
    return x is True or x in ('True', 'true')


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Units:
    """Per-unit sums: model num/den, count of infinite records, Uniform num/den."""

    def __init__(self):
        self.u = {}

    def add(self, uid, num=0., den=0., inf=0, unum=0., uden=0.):
        a = self.u.setdefault(str(uid), [0., 0., 0, 0., 0.])
        a[0] += num; a[1] += den; a[2] += inf; a[3] += unum; a[4] += uden

    def arrays(self):
        rows = [self.u[k] for k in sorted(self.u)]
        return [np.array([r[i] for r in rows], float) for i in range(5)]


def extract(key, d):
    """Units for one target from its mode directory (the folder holding mode_result.json)."""
    A = Units()
    if key in ('multi_attribute_choice', 'false_belief_story_qa', 'politeness_pragmatics', 'pragmatic_language_interpretation'):
        for r in jsonl(d / 'responses.jsonl'):
            if key == 'multi_attribute_choice':
                inf = truthy(r['typed_choice_nll_is_infinite'])
                A.add((r['experiment'], r['participant']), 0. if inf else float(r['typed_choice_nll']), 1, int(inf), LN(len(r['output_keys'])), 1)
            elif key == 'false_belief_story_qa':
                A.add((r['experiment'], r['original_story_block']), float(truthy(r['original_answer_exact_match'])), 1, 0, 1 / len(r['options']), 1)
            else:
                w, inf = float(r['human_response_count']), truthy(r.get('human_response_nll_is_infinite'))
                A.add((r['experiment'], r['trial_id']), 0. if inf else w * float(r['human_response_mean_nll']), w, int(inf), w * LN(len(r['options'])), w)
    elif key == 'instrumental_learning':
        for f in sorted(glob.glob(str(d / 'run_[0-9]*.csv'))):
            for r in csv.DictReader(open(f)):
                v = [float(r['mean0']), float(r['mean1'])]
                A.add(r['run'], float(v[int(float(r['choice']))] == max(v)), 1, 0, sum(x == max(v) for x in v) / 2, 1)
    elif key == 'two_step':
        for f in sorted(glob.glob(str(d / 'original_runner/run_[0-9]*.csv'))):
            for r in csv.DictReader(open(f)):
                A.add(r['run'], float(r['reward']), 1, 0, sum(float(r[k]) for k in ('probsA', 'probsB', 'probsC', 'probsD')) / 4, 1)
    elif key == 'nonstationary_bandit':
        for r in csv.DictReader(open(d / 'original_runner/trials.csv')):
            A.add(r['run'], float(r['accurate']), 1)
    elif key == 'horizon:human_predictive':
        for r in jsonl(d / 'original_runner/scores/decision_scores.jsonl'):
            A.add(r['metadata']['participant'], float(r['nll']), 1)
    return A


def interval(key, A, printed_value):
    row, label, direction, percent, unit, const = TARGETS[key]
    num, den, inf, unum, uden = A.arrays()
    keep = den > 0
    num, den, inf, unum, uden = num[keep], den[keep], inf[keep], unum[keep], uden[keep]
    scale = 100. if percent else 1.
    model = math.inf if inf.sum() else num.sum() / den.sum()
    uniform = const if const is not None else unum.sum() / uden.sum()
    if printed_value is not None and not (math.isinf(model) and math.isinf(printed_value)) and \
            not math.isclose(model, printed_value, rel_tol=1e-9, abs_tol=1e-12):
        raise SystemExit(f'{key}: recomputed {model!r} differs from mode_result.json {printed_value!r}')
    better = model > uniform + 1e-12 if direction == 'higher' else model < uniform - 1e-12
    out = {'target': label, 'mode_key': key, 'direction': direction, 'unit': unit, 'units': int(len(num)),
           'model': model * scale, 'uniform': uniform * scale, 'point_better_than_uniform': bool(better)}
    if math.isinf(model):
        return {**out, 'diff': math.inf, 'lo': None, 'hi': None, 'class': 'infinite'}
    rng = np.random.default_rng([SEED, row])
    idx = rng.integers(0, len(num), size=(B, len(num)))
    m = np.where(inf[idx].sum(1) > 0, np.inf, num[idx].sum(1) / den[idx].sum(1))
    u = np.full(B, const) if const is not None else unum[idx].sum(1) / uden[idx].sum(1)
    d = (m - u) * scale
    if np.ptp(d) < 1e-12:
        return {**out, 'diff': float(d[0]), 'lo': None, 'hi': None, 'class': 'constant'}
    lo, hi = np.percentile(d, [2.5, 97.5])
    up = direction == 'higher'
    cls = 'better' if (lo > 0 if up else hi < 0) else 'worse' if (hi < 0 if up else lo > 0) else 'contains_zero'
    return {**out, 'diff': (model - uniform) * scale, 'lo': float(lo), 'hi': float(hi), 'class': cls}


def score(run_dir):
    """Score every lite target found under run_dir; return the rows and the reliably-better count."""
    run_dir = Path(run_dir)
    found = {}
    for f in sorted(run_dir.rglob('mode_result.json')):
        r = json.loads(f.read_text())
        key = r.get('key') or (r['task'] if r['mode'] == 'primary_protocol' else f"{r['task']}:{r['mode']}")
        if key in TARGETS:
            if key in found:
                raise SystemExit(f'{key} appears twice under {run_dir}: {found[key][0]} and {f}')
            found[key] = (f, r)
    rows = []
    for key in TARGETS:
        if key not in found:
            rows.append({'target': TARGETS[key][1], 'mode_key': key, 'class': 'missing'})
            continue
        f, r = found[key]
        if r.get('status') != 'complete':
            rows.append({'target': TARGETS[key][1], 'mode_key': key, 'class': 'incomplete'})
            continue
        p = r['primary']
        value = math.inf if p['value_status'] != 'finite' else float(p['value'])
        rows.append(interval(key, extract(key, f.parent), value))
    return rows, sum(r['class'] == 'better' for r in rows)
