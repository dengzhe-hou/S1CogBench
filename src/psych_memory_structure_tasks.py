"""Original Psych-101 holdout histories with an additional legal-choice readout.

This interface predicts human responses from exact, untruncated human prefixes.
Typed-choice NLL is not the original full-vocabulary NLL. Observed rewards belong
only to the source human action; no autonomous or counterfactual reward is made.
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re

from src.psych_category_tasks import CategoryTrial
from src.psych_typed import DEFAULT_SOURCE, RESPONSE

TASK_SOURCES = {
    'recent_probes': ('enkavi2019recentprobes/exp1.csv',),
    'structured_bandit': tuple(f'schulz2020finding/exp{i}.csv' for i in range(1, 6)),
    'reward_revaluation': ('tomov2021multitask/exp1.csv', 'tomov2021multitask/exp3.csv'),
}
CATALOG_TASK_IDS = {task: task for task in TASK_SOURCES}
CATALOG_TASK_IDS['reward_revaluation'] = 'multi_task_reward_revaluation'
PROTOCOL_VERSION = 'psych101-memory-structure-full-prefix-typed-v1'
TASK_NOTES = {
    'recent_probes': 'Exact human transcript with visible external history; this does not reproduce disappearance of the six-letter set or human memory timing. Unanswered displays stay in later prefixes. Membership labels are derived from the displayed set.',
    'structured_bandit': 'All five original source experiments and all 30 rounds/10 trials retained. Only the chosen human action has recorded reward; no objective best-action label or autonomous reward is inferred.',
    'reward_revaluation': 'Original exp1/exp3, all market prices, rooms, resources and human feedback retained. Some exp1 stated rewards disagree with the literal price-resource dot product; both original values and discrepancy metadata are preserved. No optimal-action label is invented.',
}
NUMBER = r'-?\d+(?:\.\d+)?'
PROBE = re.compile(r"You are shown the letters \[('[A-Z]'(?:, '[A-Z]'){5})\]\. You see the letter ([A-Z])\.")
ROUND = re.compile(r'You are playing round (\d+):')
PRICES = re.compile(r'The current market prices are (' + NUMBER + r') for wood, (' + NUMBER + r') for stone, and (' + NUMBER + r') for iron\.')


def _source(row, source_row, task):
    if row['experiment'] not in TASK_SOURCES[task]:
        raise ValueError(f'Unexpected original experiment for {task}: {row["experiment"]}')
    text = row['text']
    spans = list(RESPONSE.finditer(text))
    if not spans or len(spans) != text.count('<<') or len(spans) != text.count('>>'):
        raise ValueError(f'Unparsed original response marker at source row {source_row}.')
    return text, spans, text[:spans[0].start()]


def _trial(task, row, source_row, index, span, keys, question, stimulus,
           item_index, correct=None, feedback=None):
    label = span.group(1)
    if label not in keys:
        raise ValueError(f'Original response outside instruction-defined actions: row={source_row}, response={index}, label={label}')
    return CategoryTrial(task, row['experiment'], row['participant'], source_row,
                         index, 'original_session', task + '_choice', item_index,
                         span.start(), row['text'][:span.start()], keys, label,
                         correct, question, json.dumps(stimulus, ensure_ascii=False), feedback)


def parse_recent_probes(row, source_row):
    text, spans, initial = _source(row, source_row, 'recent_probes')
    mapping = re.search(r'If you think it was, you have to press ([A-Z])\. If you think it was not, press ([A-Z])\.', initial)
    if mapping is None or mapping.group(1) == mapping.group(2):
        raise ValueError('Recent-probes actions must be stated in the original instructions.')
    keys = mapping.groups()
    question = {'type': 'choice', 'instructions': 'Was the probe letter part of the immediately preceding six-letter set?',
                'criteria': {keys[0]: 'The probe was in that six-letter set.', keys[1]: 'The probe was not in that six-letter set.'}}
    displays = list(PROBE.finditer(text))
    if len(displays) != text.count('You are shown the letters '):
        raise ValueError('Every original six-letter display must parse, including displays without a response.')
    trials, display_index = [], 0
    for index, span in enumerate(spans):
        while display_index + 1 < len(displays) and displays[display_index + 1].start() < span.start():
            display_index += 1
        display = displays[display_index]
        if text[display.end():span.start()] != ' You press ':
            raise ValueError('Recent-probes response must follow its current original stimulus.')
        letters = re.findall(r"'([A-Z])'", display.group(1))
        probe = display.group(2)
        previous = re.findall(r"'([A-Z])'", displays[display_index - 1].group(1)) if display_index else None
        member = probe in letters
        stimulus = {'display_index': display_index, 'letters': letters, 'probe': probe,
                    'previous_display_letters': previous, 'is_current_member': member,
                    'is_previous_display_member': probe in previous if previous is not None else None,
                    'recent_negative': (not member and probe in previous) if previous is not None else None,
                    'external_history_support': True}
        trials.append(_trial('recent_probes', row, source_row, index, span, keys, question,
                             stimulus, display_index, correct=keys[0] if member else keys[1]))
    return trials


def parse_structured_bandit(row, source_row):
    text, spans, initial = _source(row, source_row, 'structured_bandit')
    mapping = re.search(r'You can choose between options (\d+(?:, \d+)* and \d+) by pressing the corresponding key\.', initial)
    rounds_match = re.search(r'You will be playing a game for (\d+) rounds\.', initial)
    trials_match = re.search(r'Each round contains (\d+) trials\.', initial)
    if mapping is None or rounds_match is None or trials_match is None:
        raise ValueError('Structured-bandit actions and rounds must come from original instructions.')
    keys = tuple(re.findall(r'\d+', mapping.group(1)))
    if len(keys) != len(set(keys)):
        raise ValueError('Repeated option in original structured-bandit instructions.')
    question = {'type': 'choice', 'instructions': 'Which option do you choose on the current trial to maximize your reward?',
                'criteria': {key: f'Choose option {key} at its original position.' for key in keys}}
    rounds = list(ROUND.finditer(text))
    if [int(x.group(1)) for x in rounds] != list(range(1, int(rounds_match.group(1)) + 1)):
        raise ValueError('Keep the complete original sequence of bandit rounds.')
    counts, result, round_index = Counter(), [], 0
    for index, span in enumerate(spans):
        while round_index + 1 < len(rounds) and rounds[round_index + 1].start() < span.start():
            round_index += 1
        if not text[:span.start()].endswith('You press ') or span.start() < rounds[round_index].end():
            raise ValueError('Unexpected structured-bandit response boundary.')
        feedback = re.match(r' and get (' + NUMBER + r') points\.', text[span.end():])
        if feedback is None:
            raise ValueError('Missing original chosen-option reward; no reward may be generated.')
        number = int(rounds[round_index].group(1))
        result.append(_trial('structured_bandit', row, source_row, index, span, keys, question,
                             {'round_number': number, 'trial_in_round': counts[number],
                              'structure_variant_source': row['experiment'], 'options_reset_each_round': True},
                             counts[number], feedback={'human_chosen_option_reward': float(feedback.group(1)),
                                                       'reward_text': feedback.group(1), 'raw': feedback.group(0)}))
        counts[number] += 1
    if any(counts[n] != int(trials_match.group(1)) for n in range(1, len(rounds) + 1)):
        raise ValueError('Every original bandit round must retain its instructed number of trials.')
    return result


def parse_reward_revaluation(row, source_row):
    text, spans, initial = _source(row, source_row, 'reward_revaluation')
    mapping = re.search(r'The doors are labeled ([A-Z]), ([A-Z]), and ([A-Z])\.', initial)
    if mapping is None or len(set(mapping.groups())) != 3:
        raise ValueError('All three door labels must be stated in the original instructions.')
    keys = mapping.groups()
    question = {'type': 'choice', 'instructions': 'Which door do you choose from the current room to maximize cumulative reward at the current market prices?',
                'criteria': {key: f'Choose the door labeled {key}.' for key in keys}}
    prices = list(PRICES.finditer(text))
    if not prices or len(prices) != text.count('The current market prices are '):
        raise ValueError('Every original market-price change must parse.')
    results, round_index, counts = [], 0, Counter()
    for index, span in enumerate(spans):
        while round_index + 1 < len(prices) and prices[round_index + 1].start() < span.start():
            round_index += 1
        room = re.search(r'You are in room (\d+)\. You press $', text[:span.start()])
        if room is None or span.start() < prices[round_index].end():
            raise ValueError('Every door response needs its current room and original market prices.')
        reward = re.match(r' and you find (' + NUMBER + r') wood, (' + NUMBER + r') stone, and (' + NUMBER + r') iron\. You get (' + NUMBER + r') points\.', text[span.end():])
        if reward is None:
            raise ValueError('Missing original resource/reward feedback.')
        price_values = prices[round_index].groups()
        resources = reward.groups()[:3]
        literal_dot_product = sum(Decimal(p) * Decimal(r) for p, r in zip(price_values, resources))
        stimulus = {'round_index': round_index, 'decision_in_round': counts[round_index],
                    'room': int(room.group(1)), 'resource_order': ['wood', 'stone', 'iron'],
                    'market_prices': [float(p) for p in price_values]}
        feedback = {'human_observed_resources': dict(zip(['wood', 'stone', 'iron'], map(float, resources))),
                    'human_chosen_door_reward': float(reward.group(4)), 'reward_text': reward.group(4),
                    'raw': reward.group(0), 'literal_price_resource_dot_product': float(literal_dot_product),
                    'reported_reward_matches_literal_dot_product': literal_dot_product == Decimal(reward.group(4)),
                    'original_reported_reward_preserved': True}
        results.append(_trial('reward_revaluation', row, source_row, index, span, keys, question,
                              stimulus, counts[round_index], feedback=feedback))
        counts[round_index] += 1
    if len(results) != 202 or len(prices) != 101 or any(n != 2 for n in counts.values()):
        raise ValueError('Expected all 101 original market-price rounds with two responses each.')
    return results


PARSERS = {'recent_probes': parse_recent_probes, 'structured_bandit': parse_structured_bandit,
           'reward_revaluation': parse_reward_revaluation}


def load_trials(task, source=DEFAULT_SOURCE):
    """Read every marked response from the official task sources, without resplitting."""
    parser = PARSERS[task]
    result = []
    with Path(source).open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] in TASK_SOURCES[task]:
                result.extend(parser(row, source_row))
    if not result:
        raise ValueError(f'No original held-out source sessions for {task}.')
    return result


def audit_task(task, source=DEFAULT_SOURCE):
    trials = load_trials(task, source)
    session_records = {}
    sources = {}
    with Path(source).open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] in TASK_SOURCES[task]:
                sources[index] = row
    for trial in trials:
        row = sources[trial.source_row]
        if trial.state != row['text'][:trial.prefix_end] or trial.state.count('<<') != trial.trial_index:
            raise AssertionError('Exact human prefix or original response ordering changed.')
        session = session_records.setdefault(trial.session_id, {'experiment': trial.experiment,
            'participant': trial.participant, 'source_row': trial.source_row, 'responses': 0,
            'original_marked_responses': row['text'].count('<<'), 'output_keys': list(trial.output_keys),
            'maximum_prefix_characters': 0})
        session['responses'] += 1
        session['maximum_prefix_characters'] = max(session['maximum_prefix_characters'], len(trial.state))
    for session in session_records.values():
        if session['responses'] != session['original_marked_responses']:
            raise AssertionError('An original marked response was lost.')
    unresponded = sum(len(PROBE.findall(r['text'])) - r['text'].count('<<') for r in sources.values()) if task == 'recent_probes' else 0
    discrepancies = [{'id': t.id, 'experiment': t.experiment, 'participant': t.participant,
                      'source_row': t.source_row, 'trial_index': t.trial_index,
                      'stimulus': json.loads(t.stimulus), 'original_feedback': t.source_feedback}
                     for t in trials if task == 'reward_revaluation' and
                     not t.source_feedback['reported_reward_matches_literal_dot_product']]
    return {'status': 'complete_interface_parse', 'task': task, 'catalog_task_id': CATALOG_TASK_IDS[task],
            'protocol_version': PROTOCOL_VERSION, 'source': str(source),
            'split': 'Original Psych-101-test official participant holdout; no new split.',
            'experiments': dict(Counter(t.experiment for t in trials)), 'sessions': len(session_records),
            'responses': len(trials), 'session_records': session_records,
            'original_unresponded_displays_retained_in_text': unresponded,
            'minimum_prefix_characters': min(len(t.state) for t in trials),
            'maximum_prefix_characters': max(len(t.state) for t in trials),
            'objective_labels': sum(t.correct_category is not None for t in trials),
            'all_markers_parsed': True, 'exact_prefix_and_response_order_checked': True,
            'truncated': False, 'generated_rewards': False,
            'task_notes': TASK_NOTES[task],
            'source_reward_literal_discrepancies': discrepancies,
            'source_reward_literal_discrepancy_count': len(discrepancies),
            'original_full_vocabulary_nll': 'Not run; the added typed readout has a different question and legal-action normalization.',
            'token_length_status': 'Character lengths are exact. Model-specific complete prompt token budgets must be checked without truncation by the shared runner.',
            'rating_semantics': 'No ordinal rating response in these three original sources.'}, trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--audit-models', nargs='*', default=[])
    args = parser.parse_args()
    args.audit_output.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in TASK_SOURCES:
        report, trials = audit_task(task, args.source)
        if args.audit_models:
            from src.psych_typed import audit_lengths
            report['model_lengths'] = {}
            for model in args.audit_models:
                length = audit_lengths(trials, [model])[model]
                length['admission_status'] = ('unsupported_full_prefix_at_current_context' if
                                             length['responses_exceeding_current_context'] else 'full_prefix_length_supported')
                report['model_lengths'][model] = length
                print(json.dumps({'task': task, 'model': model, 'length_audit': length}), flush=True)
            report['token_length_status'] = 'Every actual complete model input tokenized on CPU, without truncation or model weights.'
        (args.audit_output / f'{task}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        results[task] = {k: report[k] for k in ['status', 'sessions', 'responses', 'maximum_prefix_characters', 'objective_labels']}
    (args.audit_output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
