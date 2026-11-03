"""An added compact presentation of complete Psych-101 human histories.

Only the visible state changes. Original held-out rows, response identities,
choices, feedback and scoring semantics are retained by each task formatter.
"""
import importlib

from src import psych_extended
from src.psych_typed import DEFAULT_SOURCE

PROTOCOL_VERSION = 'psych101-compact-full-history-typed-v1'
TASK_MODULES = {
    'multiple_cue_judgment': 'src.psych_cue_compact',
    'reward_revaluation': 'src.psych_reward_compact',
    'columbia_card_task': 'src.psych_columbia_compact',
    'risky_choice_information_search': 'src.psych_search_compact',
}
ORIGINAL_MODULES = {
    'columbia_card_task': 'src.psych_columbia',
    'risky_choice_information_search': 'src.psych_risk_tasks',
}
HISTORY = (
    'Complete original human history in a compact presentation, with all prior '
    'responses and feedback in original order; teacher-forced; no truncation.'
)
PRESENTATION_NOTE = (
    'Added compact presentation protocol, not the official original wording. '
    'Original full-text evaluation remains a separate protocol.'
)
PREFIX_OFFSET_NOTE = (
    'prefix_end_character / source_prefix_chars refer to the original source '
    'text offset before the current response marker, not to the compact state.'
)


def load_trials(task, source=DEFAULT_SOURCE):
    return importlib.import_module(TASK_MODULES[task]).load_trials(source)


def protocol_metadata():
    return {
        'protocol_version': PROTOCOL_VERSION,
        'presentation_protocol': PROTOCOL_VERSION,
        'presentation_note': PRESENTATION_NOTE,
        'prefix_offset_semantics': PREFIX_OFFSET_NOTE,
        'history': HISTORY,
        'history_mode': 'teacher_forced_complete_human_history',
        'truncation': False,
    }


def inventory(trials):
    module = importlib.import_module(ORIGINAL_MODULES[trials[0].task]) if trials[0].task in ORIGINAL_MODULES else psych_extended
    result = module.inventory(trials)
    result.update(protocol_metadata())
    result.update(
        compact_state_chars_max=max(len(t.state) for t in trials),
        compact_state_chars_sum=sum(len(t.state) for t in trials),
        source_prefix_chars_max=max(t.prefix_end for t in trials),
        source_prefix_chars_sum=sum(t.prefix_end for t in trials),
    )
    return result


def score(trial, prediction):
    module = importlib.import_module(ORIGINAL_MODULES[trial.task]) if trial.task in ORIGINAL_MODULES else psych_extended
    result = module.score(trial, prediction)
    result.update(protocol_metadata())
    result.update(compact_state_chars=len(trial.state), source_prefix_chars=trial.prefix_end)
    return result


def summarize(rows, expected):
    task = rows[0]['task'] if rows else None
    module = importlib.import_module(ORIGINAL_MODULES[task]) if task in ORIGINAL_MODULES else psych_extended
    result = module.summarize(rows, expected)
    result.update(protocol_metadata())
    return result
