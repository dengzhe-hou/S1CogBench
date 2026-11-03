"""Original Columbia Card Task human-history prediction with a typed readout.

All original human rounds, choices, feedback and response coordinates are kept.
The model never receives the current answer/feedback. This adapter supplies no
counterfactual card environment and does not score model-generated returns.
"""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from src.psych_category_tasks import CategoryTrial, summarize_category
from src.psych_typed import DEFAULT_SOURCE, RESPONSE, score_prediction

TASK = 'columbia_card_task'
EXPERIMENT = 'frey2017cct/exp1.csv'
PROTOCOL_VERSION = 'psych101-columbia-full-prefix-typed-v1'
ROUND = re.compile(r'Round (\d+):\n'
                   r'You will be awarded (\d+) points for turning over a gain card\.\n'
                   r'You will lose (\d+) points for turning over a loss card\.\n'
                   r'There are (\d+) loss cards in this round\.\n')
TURN = re.compile(r' and turn over a (gain|loss) card\. Your current score is (-?\d+)\.'
                  r'( The round has now ended because you encountered a loss card\.)?')
STOP = ' and claim your payout.'
FINAL = re.compile(r'Your final score for this round is (-?\d+)\.')


def parse_session(row, source_row):
    if row['experiment'] != EXPERIMENT:
        raise ValueError('This adapter implements only the original Frey Columbia source.')
    text = row['text']
    responses = list(RESPONSE.finditer(text))
    if not responses or len(responses) != text.count('<<') or len(responses) != text.count('>>'):
        raise ValueError('Every original response marker must parse exactly once.')
    initial = text[:responses[0].start()]
    mapping = re.search(r'Press ([A-Z]) to turn a card over, or ([A-Z]) to stop the round and claim your current payout\.', initial)
    declared_rounds = re.search(r'You will play a games with (\d+) rounds\.', initial)
    cards = re.search(r'you will be presented with (\d+) face-down cards\.', initial)
    if mapping is None or declared_rounds is None or cards is None or len(set(mapping.groups())) != 2:
        raise ValueError('Actions, deck size and number of rounds must come from original instructions.')
    keys, n_rounds, n_cards = mapping.groups(), int(declared_rounds.group(1)), int(cards.group(1))
    rounds = list(ROUND.finditer(text))
    if [int(m.group(1)) for m in rounds] != list(range(1, n_rounds + 1)):
        raise ValueError('All declared rounds and original risk parameters must be retained.')
    question = {'type': 'choice', 'instructions': 'Do you turn over another card or stop the current round and claim your payout?',
                'criteria': {keys[0]: 'Turn over another card.', keys[1]: 'Stop this round and claim the current payout.'}}
    result, cursor = [], 0
    for ri, round_match in enumerate(rounds):
        end = rounds[ri + 1].start() if ri + 1 < len(rounds) else len(text)
        number, gain, loss, loss_cards = map(int, round_match.groups())
        final = list(FINAL.finditer(text, round_match.end(), end))
        if len(final) != 1 or not 0 < loss_cards <= n_cards:
            raise ValueError('Each source round needs its original terminal score and valid deck parameters.')
        current_score, turned, decision, ended = 0, 0, 0, False
        while cursor < len(responses) and responses[cursor].start() < end:
            span = responses[cursor]
            if ended or span.start() < round_match.end() or not text[:span.start()].endswith('You press '):
                raise ValueError('Response must precede round termination and follow its original decision cue.')
            label = span.group(1)
            if label not in keys or turned >= n_cards:
                raise ValueError('Source action is outside the original legal actions.')
            suffix = text[span.end():end].split('\n', 1)[0]
            visible = {'round_number': number, 'decision_in_round': decision,
                       'deck_cards': n_cards, 'gain_amount': gain, 'loss_amount': loss,
                       'loss_card_count': loss_cards, 'gain_cards_revealed_before': turned,
                       'cards_remaining_before': n_cards - turned,
                       'current_human_score_before': current_score}
            if label == keys[0]:
                outcome = TURN.fullmatch(suffix)
                if outcome is None or bool(outcome.group(3)) != (outcome.group(1) == 'loss'):
                    raise ValueError('Original card outcome and terminal feedback must parse.')
                card, score_after = outcome.group(1), int(outcome.group(2))
                expected_score = current_score + (gain if card == 'gain' else -loss)
                ended = card == 'loss'
                feedback = {'human_action': 'turn', 'human_card_outcome': card,
                            'human_score_after': score_after, 'raw': suffix,
                            'arithmetic_expected_score': expected_score,
                            'original_score_arithmetic_disagrees': score_after != expected_score}
                turned += 1
            else:
                if suffix != STOP:
                    raise ValueError('Original stop feedback must remain unchanged.')
                score_after, ended = current_score, True
                feedback = {'human_action': 'stop', 'human_card_outcome': None,
                            'human_score_after': score_after, 'raw': suffix,
                            'original_score_arithmetic_disagrees': False}
            feedback['round_ended_after_human_action'] = ended
            if ended:
                feedback['human_terminal_score'] = int(final[0].group(1))
                feedback['original_terminal_score_disagrees'] = int(final[0].group(1)) != score_after
            result.append(CategoryTrial(
                TASK, row['experiment'], row['participant'], source_row, cursor,
                'original_session', 'turn_or_stop', decision, span.start(), text[:span.start()],
                keys, label, None, question, json.dumps(visible, ensure_ascii=False), feedback))
            current_score, decision, cursor = score_after, decision + 1, cursor + 1
        if not ended or decision == 0:
            raise ValueError('Do not omit an unfinished or empty original round.')
    if cursor != len(responses):
        raise ValueError('Every original response must be retained in source order.')
    return result


def load_trials(source=DEFAULT_SOURCE):
    trials = []
    with Path(source).open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] == EXPERIMENT:
                trials.extend(parse_session(row, source_row))
    if not trials:
        raise ValueError('No original Columbia sessions in the requested source.')
    return trials


def inventory(trials):
    sessions, outcomes, conditions = {}, Counter(), Counter()
    for t in trials:
        visible = json.loads(t.stimulus)
        session = sessions.setdefault(t.session_id, {'participant': t.participant, 'source_row': t.source_row,
            'responses': 0, 'rounds': 0, 'output_keys': list(t.output_keys),
            'first_response_id': t.id, 'last_response_id': t.id})
        session['responses'] += 1
        session['last_response_id'] = t.id
        session['rounds'] += visible['decision_in_round'] == 0
        if visible['decision_in_round'] == 0:
            conditions[(visible['gain_amount'], visible['loss_amount'], visible['loss_card_count'])] += 1
        feedback = t.source_feedback
        outcomes[feedback['human_card_outcome'] or 'stop'] += 1
    return {'task': TASK, 'experiment': EXPERIMENT, 'protocol_version': PROTOCOL_VERSION,
            'sessions': len(sessions), 'participants': len({str(t.participant) for t in trials}),
            'responses': len(trials), 'rounds': sum(s['rounds'] for s in sessions.values()),
            'session_counts': sessions, 'human_observed_action_outcomes': dict(outcomes),
            'risk_conditions': [{'gain_amount': k[0], 'loss_amount': k[1], 'loss_card_count': k[2],
                                 'source_rounds': v} for k,v in sorted(conditions.items())],
            'original_score_arithmetic_disagreements': sum(t.source_feedback['original_score_arithmetic_disagrees'] for t in trials),
            'original_terminal_score_disagreements': sum(t.source_feedback.get('original_terminal_score_disagrees', False) for t in trials),
            'minimum_full_prefix_characters': min(t.prefix_end for t in trials),
            'maximum_full_prefix_characters': max(t.prefix_end for t in trials),
            'split': 'Original Psych-101-test participant holdout; no resplitting.',
            'history': 'Exact full original prefix through the current decision cue, all past human choices and feedback; no truncation or round reset.',
            'evaluation': 'Human-history action prediction, not autonomous reward.',
            'original_protocol_nll_status': 'not_run_by_this_typed_runner',
            'objective_action_accuracy': None,
            'score_note': 'Neither an observed favorable card nor a human choice defines the objectively correct action.'}


def score(trial, prediction):
    row = score_prediction(trial, prediction)
    row['category_correct_under_human_history'] = None
    row['native_rating_absolute_error'] = None
    row.update(json.loads(trial.stimulus))
    row['model_turn_probability'] = row['probabilities'][trial.output_keys[0]]
    row['model_stop_probability'] = row['probabilities'][trial.output_keys[1]]
    row['model_action'] = ('turn' if row['choice'] == trial.output_keys[0] else
                           'stop' if row['choice'] == trial.output_keys[1] else 'invalid')
    row['human_action'] = trial.source_feedback['human_action']
    row['model_closed_loop_reward'] = None
    return row


def summarize(rows, expected):
    result = summarize_category(rows, expected)
    durations = [r['batch_elapsed_seconds'] for r in rows]
    ordered = sorted(durations)
    seconds = sum(durations)
    result.update(task=TASK, protocol_version=PROTOCOL_VERSION,
                  completed_batch_wall_seconds=seconds,
                  shared_responses_per_second=len(rows) / seconds if seconds else None,
                  latency_p50_seconds=statistics.median(durations) if durations else None,
                  latency_p95_seconds=ordered[-(-95 * len(ordered) // 100) - 1] if ordered else None,
                  latency_scope='Original scalar calls on shared device, including first call; not human reaction time.',
                  objective_action_accuracy=None, model_closed_loop_reward=None)
    return result
