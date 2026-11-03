"""`eval --smoke`: a few adapter calls per mode, checked against stored values.

Every mode runs its original runner with the adapter under a call budget.
The run stops after the budget (temporal discounting completes, since its
13-question original questionnaire catches every exception). The record of
each call (state length, legal keys, native choice, reported value) and the
scores of the scored items are compared with smoke_expected.json, which was
produced with examples/uniform_reference_agent.py on the inputs of the
2026-10-26 CPU validation. The expected values hold no participant response.
"""
import json
import math
from pathlib import Path

LIMIT = 5
LIMITS = {'temporal_discounting': 1000}
EXPECTED = Path(__file__).with_name('smoke_expected.json')
STATE = {'limit': 0, 'trace': None, 'adapter': None, 'agent': None}


class SmokeStop(BaseException):
    """Ends a run after its call budget. A BaseException, so the original
    runners' `except Exception` failure handlers do not treat it as a model failure."""


class BudgetAgent:
    """Forwards every call to the adapter and stops after `limit` calls."""

    def __init__(self, inner, limit, trace):
        self._inner, self._limit, self._trace = inner, limit, trace

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _before(self):
        if len(self._trace) >= self._limit:
            raise SmokeStop()

    def _record(self, method, state, keys=None, value=None):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
        row = {'method': method, 'state_chars': len(text)}
        if keys is not None:
            keys = list(keys)
            row['keys'] = keys if len(keys) <= 12 else [len(keys), keys[0], keys[-1]]
            row['native_choice'] = getattr(self._inner, 'last_native_choice', None)
        else:
            row['value'] = None if value is None else float(value)
        self._trace.append(row)

    def choice_probabilities(self, state, question, output_keys):
        self._before()
        p = self._inner.choice_probabilities(state, question, output_keys)
        self._record('choice_probabilities', state, keys=output_keys)
        return p

    def posterior_probability(self, state, question):
        self._before()
        p = self._inner.posterior_probability(state, question)
        self._record('posterior_probability', state, value=p)
        return p

    def confidence_probability(self, state, chosen, specifications):
        self._before()
        q = self._inner.confidence_probability(state, chosen, specifications)
        self._record('confidence_probability', state, value=q)
        return q


def configure(adapter, limit, trace):
    STATE.update(adapter=adapter, limit=limit, trace=trace, agent=None)


def budget_agent(device):
    """The adapter loaded once per worker, wrapped with the shared call budget."""
    if STATE['agent'] is None:
        from src.paper_horizon import load_agent
        STATE['agent'] = load_agent(STATE['adapter'], device)
    return BudgetAgent(STATE['agent'], STATE['limit'], STATE['trace'])


def observed(result, summary_values):
    return {'status': result['status'], 'calls': result.get('smoke_trace', []),
            'scores': summary_values}


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def compare(key, value, expected):
    """Return a list of human-readable differences for one mode."""
    if key not in expected:
        return [f'{key}: no stored expected value']
    want = expected[key]
    problems = []
    if value['status'] != want['status']:
        problems.append(f"{key}: status {value['status']} != expected {want['status']}")
    for i, (got, ref) in enumerate(zip(value['calls'], want['calls'])):
        if not close(got, ref):
            problems.append(f'{key}: call {i} {got} != expected {ref}')
            break
    if len(value['calls']) != len(want['calls']):
        problems.append(f"{key}: {len(value['calls'])} calls != expected {len(want['calls'])}")
    if not close(value['scores'], want['scores']):
        problems.append(f"{key}: scores {value['scores']} != expected {want['scores']}")
    return problems
