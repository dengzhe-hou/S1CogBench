"""Additional, complete compact-history presentation of multiple-cue judgment.

Only the representation of already visible past trials changes. The original
instructions, current unanswered cue line, all source/scoring metadata and the
original full-text interface are preserved.
"""
from dataclasses import replace
import re

from src.psych_cue_tasks import load_trials as load_original_trials
from src.psych_typed import DEFAULT_SOURCE

PROTOCOL_VERSION = 'psych101-multiple-cue-complete-compact-history-v1'
HISTORY_LEGEND = "Past trials in chronological order. 'none' means no feedback was supplied.\n"
TABLE_HEADER = 'Trial|Progladine|Amalydine|Your estimate of Caldionine|Feedback|Correct Caldionine\n'
CURRENT_HEADER = '\nCurrent trial:\n'

_COMPLETED = re.compile(
    r'Progladine: (?P<progladine>[^.\n]+)\. Amalydine: (?P<amalydine>[^.\n]+)\. '
    r'You say that the Caldionine concentration is <<(?P<response>[^<>\n]+)>>\.'
    r'(?: That is (?P<feedback>correct|incorrect)\. The correct concentration of Caldionine is '
    r'(?:indeed )?(?P<target>[^.\n]+)\.)? *')
_CURRENT = re.compile(
    r'Progladine: [^.\n]+\. Amalydine: [^.\n]+\. '
    r'You say that the Caldionine concentration is ')


def compact_state(state):
    """Encode one original visible prefix; never consult response metadata."""
    start = re.search(r'^Progladine:', state, flags=re.MULTILINE)
    if start is None:
        raise ValueError('Expected the first original multiple-cue trial after its instructions.')
    instructions = state[:start.start()]
    lines = state[start.start():].splitlines()
    if not lines or _CURRENT.fullmatch(lines[-1]) is None:
        raise ValueError('The prefix must end at the original unanswered current trial.')
    history = []
    for index, line in enumerate(lines[:-1], start=1):
        event = _COMPLETED.fullmatch(line)
        if event is None:
            raise ValueError('Unparsed source body line; compact conversion cannot omit instructions or events.')
        fields = event.groupdict()
        history.append('|'.join((str(index), fields['progladine'], fields['amalydine'],
            fields['response'], fields['feedback'] or 'no feedback', fields['target'] or 'none')))
    past = ''.join(line + '\n' for line in history)
    return instructions + HISTORY_LEGEND + TABLE_HEADER + past + CURRENT_HEADER + lines[-1]


def compact_trial(trial):
    if trial.task != 'multiple_cue_judgment':
        raise ValueError('This additional presentation only implements multiple-cue judgment.')
    return replace(trial, state=compact_state(trial.state))


def load_trials(source=DEFAULT_SOURCE):
    return [compact_trial(trial) for trial in load_original_trials('multiple_cue_judgment', source)]
