"""Shared scoring for additional complete Psych-101 human-history tasks."""
from collections import Counter
import importlib
import math
import statistics

from src.psych_typed import DEFAULT_SOURCE, score_prediction
from src.psych_category_tasks import summarize_category

PROTOCOL_VERSION = 'psych101-extended-full-prefix-typed-v1'
TASK_MODULES = {
    **dict.fromkeys(('recent_probes', 'structured_bandit', 'reward_revaluation'),
                    'src.psych_memory_structure_tasks'),
    **dict.fromkeys(('multiple_cue_judgment', 'multi_attribute_choice'),
                    'src.psych_cue_tasks'),
}


def task_module(task):
    return importlib.import_module(TASK_MODULES[task])


def load_trials(task, source=DEFAULT_SOURCE):
    return task_module(task).load_trials(task, source)


def inventory(trials):
    sessions = {}
    for t in trials:
        session = sessions.setdefault(t.session_id, {'experiment': t.experiment,
            'participant': t.participant, 'source_row': t.source_row, 'responses': 0})
        session['responses'] += 1
    task = trials[0].task
    notes = getattr(task_module(task), 'TASK_NOTES', {})
    return {'task': task, 'protocol_version': PROTOCOL_VERSION,
        'sessions': len(sessions), 'responses': len(trials), 'session_counts': sessions,
        'response_kinds': dict(Counter(t.phase + '/' + t.response_kind for t in trials)),
        'history': 'Exact original prefix before current response, all prior human choices and feedback; no truncation.',
        'split': 'Original Psych-101-test participant holdout; no resplitting.',
        'objective_scored_responses': sum(t.correct_category is not None for t in trials),
        'task_notes': notes.get(task),
        'interpretation': 'Predicting original human responses; no autonomous or counterfactual reward.',
        'original_protocol_nll_status': 'not_run_by_this_typed_runner'}


def score(trial, prediction):
    module = task_module(trial.task)
    if hasattr(module, 'score'):
        row = module.score(trial, prediction)
    else:
        row = score_prediction(trial, prediction)
    if trial.correct_category is None:
        row['category_correct_under_human_history'] = None
    row.setdefault('native_rating_absolute_error', None)
    return row


def summarize(rows, expected):
    result = summarize_category(rows, expected)
    # Every formal call uses the original scalar adapter. Include all calls,
    # including first-call overhead; these are shared-device observed latencies.
    seconds = sorted(r['batch_elapsed_seconds'] for r in rows)
    result.update(latency_unit='seconds', latency_scope='all original scalar calls on shared device; includes first call',
        latency_p50_seconds=statistics.median(seconds) if seconds else None,
        latency_p95_seconds=seconds[math.ceil(.95 * len(seconds)) - 1] if seconds else None)
    return result
