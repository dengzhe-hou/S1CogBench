"""Model-independent Callaway 2022 Exp1 human reveal/STOP prediction task.

Retains the original labels, legal actions, observed-history boundary and folds.
The natural-language rendering is the existing benchmark's adapted interface.
"""
from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mouselab_callaway2022_official'
GRAPH = {0: [1, 5, 9, 13], 1: [2], 2: [3, 4], 3: [], 4: [],
         5: [6], 6: [7, 8], 7: [], 8: [], 9: [10], 10: [11, 12],
         11: [], 12: [], 13: [14], 14: [15, 16], 15: [], 16: []}
PARENT = {child: parent for parent, children in GRAPH.items() for child in children}
RULES = (
    'Plan a route from root 0 to a leaf to maximize your score, the sum of the 3 visited node values. '
    'Each hidden node independently has value -10, -5, +5, or +10 with equal probability. '
    'Root 0 has known value 0.\n'
    'Tree (parent -> children): 0 -> 1,5,9,13; 1 -> 2; 2 -> 3,4; '
    '5 -> 6; 6 -> 7,8; 9 -> 10; 10 -> 11,12; 13 -> 14; 14 -> 15,16.\n'
    'Before moving, reveal a hidden node only if its parent is known. Each reveal requires '
    'a 3-second wait and deducts no points. STOP ends revealing: then choose and travel a '
    'route, with no further reveals.\n'
)


def legal_actions(history):
    known = {0, *(int(node) for node, _ in history)}
    return ['STOP'] + [f'R{node}' for node in range(1, 17)
                       if node not in known and PARENT[node] in known]


def render_request(history):
    # Only the already observed prefix is accepted, never the full trial record.
    actions = legal_actions(history)
    observed = '; '.join(f'{int(node)}={int(value):+d}' for node, value in history) or 'none'
    state = RULES + 'Observed reveals in order: ' + observed + '.'
    question = {'type': 'choice', 'instructions': 'Choose your next action in this task.',
                'criteria': {key: ('Stop revealing and choose a route.' if key == 'STOP'
                                  else f'Reveal node {key[1:]} (wait 3 seconds).')
                             for key in actions}}
    return state, question, actions


def build_records():
    data = json.loads((DATA / 'trials.json').read_text())
    folds = json.loads((DATA / 'folds.json').read_text())['folds']
    by_trial = {i: f['fold'] for f in folds for i in f['test']}
    assert sorted(by_trial) == list(range(1, 26))
    assert len(data) == 95 and all(len(trials) == 25 for trials in data.values())
    records = []
    for wid, trials in data.items():
        for trial_index, trial in enumerate(trials, 1):
            assert trial['mdp'] == '4O8nSiCFkbB' and str(trial['graph']) == '412'
            history = []
            for step, target in enumerate([f'R{int(n)}' for n, _ in trial['reveals']] + ['STOP']):
                state, question, actions = render_request(history)
                assert target in actions, (wid, trial_index, step, target, actions)
                record = {'wid': wid, 'trial': trial_index, 'decision': step + 1,
                          'fold': by_trial[trial_index], 'human_action': target,
                          'official_action_index': 0 if target == 'STOP' else int(target[1:]) + 1,
                          'state': state, 'question': question, 'legal_actions': actions,
                          'observed_history': [list(x) for x in history]}
                records.append(record)
                if target != 'STOP':
                    node, value = trial['reveals'][step]
                    assert value in [-10, -5, 5, 10]
                    history.append((int(node), int(value)))
    assert len(records) == 20363
    assert sum(r['human_action'] == 'STOP' for r in records) == 2375
    return records


def check_against_julia(records):
    path = ROOT / 'runs/mouselab_official_metagreedy_20261003/planning/model/results/exp1/predictions.json'
    original = json.loads(path.read_text())
    grouped_original = defaultdict(list)
    grouped_records = defaultdict(list)
    for row in original:
        grouped_original[row['wid']].append(row)
    for row in records:
        grouped_records[row['wid']].append(row)
    assert grouped_records.keys() == grouped_original.keys()
    checked = 0
    for wid, rows in grouped_records.items():
        source = grouped_original[wid]
        assert len(rows) == len(source)
        for row, baseline in zip(rows, source):
            assert row['official_action_index'] == baseline['c']
            expected_indices = {0 if a == 'STOP' else int(a[1:]) + 1 for a in row['legal_actions']}
            source_indices = {i for i, p in enumerate(baseline['predictions']['MetaGreedy']) if p > 0}
            assert expected_indices == source_indices, (wid, row['trial'], row['decision'])
            checked += 1
    return {'checked_decisions': checked, 'official_labels_equal': True,
            'legal_support_equal': True, 'source': str(path)}
