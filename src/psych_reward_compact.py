"""Complete, compact reward-revaluation histories; original trials stay intact.

Only CategoryTrial.state changes. Initial instructions, every price announcement
(including repetitions), decision order and all observed feedback are preserved.
Numbers are copied as text, never recalculated; original inconsistent rewards are
retained. This is a new presentation protocol, not the original full-text NLL.
"""
from dataclasses import replace
import json
from pathlib import Path
import re

from src.psych_memory_structure_tasks import NUMBER, PRICES, TASK_SOURCES, parse_reward_revaluation
from src.psych_typed import DEFAULT_SOURCE

PROTOCOL_VERSION = 'psych101-reward-revaluation-complete-history-compact-v1'
LEGEND = (
    'Complete history in chronological order:\n'
    'P wood stone iron = a new round with these market prices, even if repeated.\n'
    'D room door wood stone iron points = the room, chosen door, resources found, and reported points.\n'
    'The final D row ends before the door choice.\n'
)
DECISION = re.compile(
    r'You are in room (?P<room>\d+)\. You press '
    r'(?P<marker><<(?P<door>[A-Z])>>) and you find '
    r'(?P<wood>' + NUMBER + r') wood, (?P<stone>' + NUMBER + r') stone, and '
    r'(?P<iron>' + NUMBER + r') iron\. You get (?P<points>' + NUMBER + r') points\.'
)
WHITESPACE = re.compile(r'\s*')


def compact_session(row, source_row):
    """Consume the full source session, failing on any unconverted body text."""
    original = parse_reward_revaluation(row, source_row)
    text = row['text']
    first_price = PRICES.search(text)
    if first_price is None:
        raise ValueError(f'No original market-price announcement at source row {source_row}.')
    # This exact slice retains all initial instructions, not a rewritten summary.
    initial = text[:first_price.start()]
    history = initial + LEGEND
    position = first_price.start()
    prices = 0
    converted = []
    while position < len(text):
        position = WHITESPACE.match(text, position).end()
        if position == len(text):
            break
        price = PRICES.match(text, position)
        if price is not None:
            history += 'P ' + ' '.join(price.groups()) + '\n'
            prices += 1
            position = price.end()
            continue
        decision = DECISION.match(text, position)
        if decision is None:
            raise ValueError(f'Unconverted source text at row={source_row}, character={position}; no sample was skipped.')
        index = len(converted)
        trial = original[index]
        if (trial.prefix_end != decision.start('marker') or
                trial.original_choice_label != decision.group('door')):
            raise ValueError(f'Original response alignment changed at row={source_row}, response={index}.')
        # Current response/feedback are appended only AFTER this input is saved.
        state = history + 'D ' + decision.group('room') + ' '
        converted.append(replace(trial, state=state))
        history += 'D ' + ' '.join(decision.group(field) for field in
                                  ['room', 'marker', 'wood', 'stone', 'iron', 'points']) + '\n'
        position = decision.end()
    if prices != 101 or len(converted) != len(original) or len(converted) != 202:
        raise ValueError(f'Incomplete original session conversion at source row {source_row}.')
    return converted


def load_trials(source=DEFAULT_SOURCE):
    """Preserve all official held-out exp1/exp3 sessions and every original ID."""
    trials = []
    with Path(source).open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] in TASK_SOURCES['reward_revaluation']:
                trials.extend(compact_session(row, source_row))
    if not trials:
        raise ValueError('No original held-out reward-revaluation sessions in the supplied source.')
    return trials
