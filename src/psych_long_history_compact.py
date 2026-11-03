"""Reversible event notation for two complete long human histories.

These presentations are not used by the evaluated modes.
Every original input prefix can be reconstructed exactly; labels and scoring do
not enter the encoding. Original full-text evaluations remain separate.
"""
from dataclasses import replace
import re

from src import psych_columbia, psych_risk_tasks
from src.psych_typed import DEFAULT_SOURCE

TASKS = ('columbia_card_task', 'risky_choice_information_search')
INTRO = 'Original instructions (unchanged):\n'
SEPARATOR = '\nComplete event log in original order:\n'
NOTES = {
    TASKS[0]: (
        'Compact notation retains every round, human key and feedback. '
        'Round N: gain, loss, loss_cards are the original round parameters. '
        'turn KEY gain/loss score=N records the human key, revealed card and resulting score; '
        'END means the loss card ended the round. stop KEY means claim payout. '
        'final N is the reported final score, retained even if inconsistent. '
        'Next key is the unanswered current decision.\n'),
    TASKS[1]: (
        'Compact notation retains every round, human gamble/color key and feedback. '
        'inspect GAMBLE,COLOR:PAYOFF records the two human responses and revealed payoff. '
        'stop GAMBLE;draw=COLOR;earn=POINTS records stopping and the observed outcome. '
        'Ball counts, inspection cost and all original rules remain visible. '
        'Next gamble is unanswered; when a gamble is already shown, only the color/stop response is unanswered.\n'),
}
# Each pair has a visible, semantic notation and an exact inverse. The original
# task parsers already reject unparsed trials; this encoder never invents events.
COLUMBIA = [
    (r'Round (\d+):\nYou will be awarded (\d+) points for turning over a gain card\.\nYou will lose (\d+) points for turning over a loss card\.\nThere are (\d+) loss cards in this round\.\n',
     r'Round (\d+): gain=(\d+),loss=(\d+),loss_cards=(\d+)\n',
     'Round {}: gain={},loss={},loss_cards={}\n',
     'Round {}:\nYou will be awarded {} points for turning over a gain card.\nYou will lose {} points for turning over a loss card.\nThere are {} loss cards in this round.\n'),
    (r'You press <<([A-Z])>> and turn over a loss card\. Your current score is (-?\d+)\. The round has now ended because you encountered a loss card\.',
     r'turn ([A-Z]) loss score=(-?\d+) END',
     'turn {} loss score={} END',
     'You press <<{}>> and turn over a loss card. Your current score is {}. The round has now ended because you encountered a loss card.'),
    (r'You press <<([A-Z])>> and turn over a gain card\. Your current score is (-?\d+)\.',
     r'turn ([A-Z]) gain score=(-?\d+)',
     'turn {} gain score={}',
     'You press <<{}>> and turn over a gain card. Your current score is {}.'),
    (r'You press <<([A-Z])>> and claim your payout\.',
     r'stop ([A-Z])', 'stop {}', 'You press <<{}>> and claim your payout.'),
    (r'Your final score for this round is (-?\d+)\.',
     r'final (-?\d+)', 'final {}', 'Your final score for this round is {}.'),
    (r'You press \Z', r'Next key: \Z', 'Next key: ', 'You press '),
]
SEARCH = [
    (r'You press <<([A-Z])>> and then type <<([a-z]+)>>\. The payoff for this combination would be (-?\d+) points\.',
     r'inspect ([A-Z]),([a-z]+):(-?\d+)',
     'inspect {},{}:{}',
     'You press <<{}>> and then type <<{}>>. The payoff for this combination would be {} points.'),
    (r'You press <<([A-Z])>> and then type <<stop>>\. A ([a-z]+) ball is chosen, and you earn (-?\d+) points\.',
     r'stop ([A-Z]);draw=([a-z]+);earn=(-?\d+)',
     'stop {};draw={};earn={}',
     'You press <<{}>> and then type <<stop>>. A {} ball is chosen, and you earn {} points.'),
    (r'You press <<([A-Z])>> and then type \Z',
     r'Gamble ([A-Z]); next color or stop: \Z',
     'Gamble {}; next color or stop: ', 'You press <<{}>> and then type '),
    (r'You press \Z', r'Next gamble: \Z', 'Next gamble: ', 'You press '),
]
PAIRS = {TASKS[0]: COLUMBIA, TASKS[1]: SEARCH}
COMPILED = {task: [(re.compile(long), re.compile(short), encoded, original)
                  for long, short, encoded, original in pairs]
            for task, pairs in PAIRS.items()}


def encode(task, prefix):
    marker = 'Round 1:\n' if task == TASKS[0] else 'A new round begins.\n'
    boundary = prefix.index(marker)
    initial, body = prefix[:boundary], prefix[boundary:]
    for long, short, encoded, original in COMPILED[task]:
        body = long.sub(lambda m: encoded.format(*m.groups()), body)
    return NOTES[task] + INTRO + initial + SEPARATOR + body


def decode(task, state):
    start = NOTES[task] + INTRO
    if not state.startswith(start):
        raise ValueError('Missing declared compact notation.')
    initial, body = state[len(start):].split(SEPARATOR, 1)
    for long, short, encoded, original in reversed(COMPILED[task]):
        body = short.sub(lambda m: original.format(*m.groups()), body)
    return initial + body


def original_trials(task, source=DEFAULT_SOURCE):
    if task == TASKS[0]:
        return psych_columbia.load_trials(source)
    return psych_risk_tasks.load_trials(task, source)


def load_trials(task, source=DEFAULT_SOURCE):
    result = []
    for trial in original_trials(task, source):
        state = encode(task, trial.state)
        if decode(task, state) != trial.state:
            raise ValueError(f'Visible original prefix does not round-trip: {trial.id}')
        result.append(replace(trial, state=state))
    return result
