"""Text adaptation of the online Fabulous Fruits task in slipstask/static/js.

The schedule and scores follow task.js/utils.ts. Images, reaction deadlines, and
the JS training score-slide one-trial offset are not reproduced.
"""

from collections import Counter
from dataclasses import dataclass
import random

FRUITS = ('apple', 'bananas', 'cherries', 'coconut', 'grape', 'kiwi',
          'lemon', 'melon', 'orange', 'pear', 'pineapple', 'strawberry')


@dataclass(frozen=True)
class Box:
    outside: str
    inside: str
    side: str


@dataclass(frozen=True)
class Trial:
    phase: str
    block: int
    box: int | None = None
    devalued: tuple[str, ...] = ()
    valued: str | None = None
    displayed: tuple[str, ...] = ()


def _devaluation_pairs(rng):
    # Port of utils.ts::soa_assign(9, 6, 3, 2): draw each box's three blocks
    # from blocks that still have capacity for two devalued boxes; restart if
    # a late box has fewer than three available blocks.
    while True:
        pairs = [[] for _ in range(9)]
        for box in range(6):
            available = [j for j, pair in enumerate(pairs) if len(pair) < 2]
            if len(available) < 3:
                break
            for block in rng.sample(available, 3):
                pairs[block].append(box)
        else:
            return [tuple(pair) for pair in pairs]


class Episode:
    def __init__(self, seed):
        self.seed = seed
        rng = random.Random(seed)
        names = list(FRUITS)
        rng.shuffle(names)
        sides = ['L'] * 3 + ['R'] * 3
        rng.shuffle(sides)
        self.boxes = tuple(Box(names[i], names[6 + i], sides[i]) for i in range(6))
        self.devaluation_pairs = _devaluation_pairs(rng)
        self.phase_order = ['SOA', 'DD']
        rng.shuffle(self.phase_order)
        trials = []
        for block in range(8):
            order = list(range(6)) * 2
            rng.shuffle(order)
            trials.extend(Trial('ID', block, box=i) for i in order)
        left = [b for b in self.boxes if b.side == 'L']
        right = [b for b in self.boxes if b.side == 'R']
        od_pairs = [(l.inside, r.inside) for l in left for r in right] * 4
        rng.shuffle(od_pairs)
        for l, r in od_pairs:
            valued, devalued = (l, r) if rng.randrange(2) == 0 else (r, l)
            shown = [valued, devalued]
            rng.shuffle(shown)
            trials.append(Trial('OD', 0, devalued=(devalued,), valued=valued,
                                displayed=tuple(shown)))
        for phase in self.phase_order:
            for block, pair in enumerate(self.devaluation_pairs):
                devalued = tuple(getattr(self.boxes[i], 'inside' if phase == 'SOA' else 'outside')
                                 for i in pair)
                order = list(range(6)) * 2
                rng.shuffle(order)
                trials.extend(Trial(phase, block, box=i, devalued=devalued) for i in order)
        self.trials = tuple(trials)

    def target(self, trial):
        if trial.phase == 'OD':
            return next(b.side for b in self.boxes if b.inside == trial.valued)
        box = self.boxes[trial.box]
        if trial.phase == 'ID':
            return box.side
        label = box.inside if trial.phase == 'SOA' else box.outside
        return 'SKIP' if label in trial.devalued else box.side

    def score(self, trial, action):
        assert action in (('L', 'R') if trial.phase in ('ID', 'OD') else ('L', 'R', 'SKIP'))
        if trial.phase == 'OD':
            return int(action == self.target(trial))
        box = self.boxes[trial.box]
        if action != box.side:
            return 0
        label = box.inside if trial.phase == 'SOA' else box.outside
        return -1 if label in trial.devalued else 1

    def feedback(self, trial, action):
        """Only training exposes fruit or empty box on each trial."""
        assert trial.phase == 'ID'
        return self.boxes[trial.box].inside if action == self.boxes[trial.box].side else None


class ObservedMemory:
    """Shared external summary computed only from the policy's own feedback."""

    def __init__(self, episode):
        outsides = tuple(b.outside for b in episode.boxes)
        self.seen = []
        self.attempts = {name: Counter() for name in outsides}
        self.empty = {name: set() for name in outsides}
        self.opened = {}

    def update(self, outside, action, fruit):
        if outside not in self.seen:
            self.seen.append(outside)
        self.attempts[outside][action] += 1
        if fruit is None:
            self.empty[outside].add(action)
        else:
            self.opened[outside] = (action, fruit)

    def inferred_side(self, outside):
        if outside in self.opened:
            return self.opened[outside][0]
        if self.empty[outside] == {'L'}:
            return 'R'
        if self.empty[outside] == {'R'}:
            return 'L'
        return None

    def text(self):
        rows = []
        for outside in self.seen:
            if outside in self.opened:
                side, fruit = self.opened[outside]
                rows.append(f'{outside}: {side} opened -> {fruit} '
                            f'({self.attempts[outside][side]} observed)')
            elif self.empty[outside]:
                side = next(iter(self.empty[outside]))
                rows.append(f'{outside}: {side} tried -> empty '
                            f'({self.attempts[outside][side]} observed)')
            else:
                rows.append(f'{outside}: no observation')
        return '; '.join(rows) if rows else 'no observations yet'


def visible_state(episode, trial, memory, previous_block_score=None):
    """Text shown before acting; never include hidden mapping or current score."""
    phase = trial.phase
    rule = ('Six boxes have different outside fruit labels. Each has one LEFT or RIGHT '
            'button that opens a different inside fruit; the other button is empty. '
            'During training, a correct opening earns +1; empty earns 0. '
            'Use only observed feedback below.')
    lines = [rule, 'Observed training feedback: ' + memory.text() + '.']
    if previous_block_score is not None:
        lines.append(f'Previous completed block score: {previous_block_score}.')
    if phase == 'ID':
        lines.append('TRAINING: each response reveals its immediate result.')
        lines.append(f'Current outside fruit: {episode.boxes[trial.box].outside}.')
    elif phase == 'OD':
        assert trial.valued is not None and len(trial.devalued) == 1
        lines.append('OUTCOME TEST: no per-trial feedback. Choose the button for the valuable inside fruit. '
                     'Correct earns +1; the other button earns 0.')
        lines.append(f'Inside fruits shown: {", ".join(trial.displayed)}. '
                     f'Spoiled inside fruit: {trial.devalued[0]}.')
    else:
        which = 'inside outcomes' if phase == 'SOA' else 'outside box labels'
        lines.append('Correctly opening a valued box earns +1; correctly opening a spoiled box '
                     'costs 1; wrong button or SKIP earns 0.')
        lines.append(f'{phase} TEST: no per-trial feedback. Spoiled {which}: '
                     + ', '.join(trial.devalued) + '.')
        lines.append(f'Current outside fruit: {episode.boxes[trial.box].outside}.')
        lines.append('Skip if opening this box would involve a spoiled fruit; otherwise open it.')
    return '\n'.join(lines)


def schedule_counts(episode):
    """Small deterministic summary used by direct correctness checks."""
    counts = Counter(t.phase for t in episode.trials)
    devalued = Counter()
    for t in episode.trials:
        if t.phase in ('SOA', 'DD') and episode.target(t) == 'SKIP':
            devalued[t.phase] += 1
    return dict(counts), dict(devalued)
