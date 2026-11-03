"""Complete Medin and weather-prediction Psych-101 held-out response protocols.

This is an added typed readout of the original full human histories. It does not
generate independent feedback or replace the official full-vocabulary scoring.
"""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics

from src.psych_typed import DEFAULT_SOURCE, RESPONSE, score_prediction

TASK_SOURCES = {
    'medin': ('levering2020revisiting/exp1.csv', 'levering2020revisiting/exp2.csv'),
    'weather': ('speekenbrink2008learning/exp1.csv',),
}
PROTOCOL_VERSION = 'psych101-category-full-prefix-typed-v1'


@dataclass(frozen=True)
class CategoryTrial:
    task: str
    experiment: str
    participant: object
    source_row: int
    trial_index: int
    phase: str
    response_kind: str
    item_index: int
    prefix_end: int
    state: str
    output_keys: tuple
    original_choice_label: str
    correct_category: object
    question: dict
    stimulus: str
    source_feedback: object

    @property
    def session_id(self):
        return f'{self.experiment}:row={self.source_row}'

    @property
    def id(self):
        return f'{self.session_id}:response={self.trial_index}'

    def identifiers(self):
        return {'id': self.id, 'task': self.task, 'experiment': self.experiment,
                'participant': self.participant, 'source_row': self.source_row,
                'session_id': self.session_id, 'session_id_origin': 'derived from original JSONL row',
                'trial_index': self.trial_index,
                'trial_id_origin': 'zero-based response position in the original session',
                'phase': self.phase, 'response_kind': self.response_kind,
                'item_index': self.item_index, 'prefix_end_character': self.prefix_end,
                'output_keys': list(self.output_keys), 'original_choice_label': self.original_choice_label,
                'correct_category_for_scoring_only': self.correct_category,
                'stimulus': self.stimulus, 'source_feedback_for_scoring_only': self.source_feedback}


def _responses(row):
    text = row['text']
    spans = list(RESPONSE.finditer(text))
    if len(spans) != text.count('<<') or len(spans) != text.count('>>') or not spans:
        raise ValueError('Source response markers must all parse exactly once.')
    return text, spans, text[:spans[0].start()]


def parse_medin(row, source_row):
    text, spans, initial = _responses(row)
    m = re.search(r'belongs to one of two categories, ([A-Z]) or ([A-Z])\.', initial)
    if m is None or m.group(1) == m.group(2):
        raise ValueError('Medin needs two original category keys in the visible instructions.')
    categories = m.groups()
    if text.count('\nTest block:\n') != 1:
        raise ValueError('Expected one original Medin learning-to-test transition.')
    boundary = text.index('\nTest block:\n')
    class_question = {'type': 'choice', 'instructions': 'Which category do you choose for the current object?',
                      'criteria': {k: f'Choose category {k}.' for k in categories}}
    rating_question = {
        'type': 'choice',
        'instructions': 'Rate how typical the current object is for the category you just chose, on the original 1 to 9 scale.',
        'criteria': {str(k): ('Rating 1: not at all typical.' if k == 1 else
                              'Rating 9: most typical.' if k == 9 else f'Rating {k} on the original 1 to 9 typicality scale.')
                     for k in range(1, 10)}}
    counts, trials = Counter(), []
    for index, span in enumerate(spans):
        prefix, label = text[:span.start()], span.group(1)
        phase = 'learning' if span.start() < boundary else 'test'
        correct = feedback = None
        if prefix.endswith(', press '):
            kind = 'category_choice'
            keys, question = categories, class_question
            stimulus = re.search(r'You see the image ([12]{3}), press $', prefix)
            if stimulus is None:
                raise ValueError('Medin category response must follow its original three-digit stimulus.')
            item_index = counts[(phase, kind)]
            if phase == 'learning':
                reward = re.match(r' and get ([01]) points\.', text[span.end():])
                if reward is None:
                    raise ValueError('Expected original 0/1 Medin feedback; do not rewrite the payoff.')
                feedback = int(reward.group(1))
                correct = label if feedback == 1 else next(k for k in categories if k != label)
        elif prefix.endswith('and judged the typicality to be '):
            kind = 'typicality_rating'
            if phase != 'test':
                raise ValueError('Typicality ratings are only defined in the original test block.')
            stimulus = re.search(r'You see the image ([12]{3}), press <<([A-Z])>> and judged the typicality to be $', prefix)
            if stimulus is None or stimulus.group(2) not in categories:
                raise ValueError('Typicality must retain the just-observed human category choice.')
            keys, question = tuple(str(k) for k in range(1, 10)), rating_question
            item_index = counts[(phase, kind)]
            if counts[(phase, 'category_choice')] != item_index + 1:
                raise ValueError('Each original test choice must be followed by its rating.')
        else:
            raise ValueError('Unrecognized Medin response field; no response is skipped.')
        if label not in keys:
            raise ValueError('Original Medin response outside the visible legal categories/ratings.')
        trials.append(CategoryTrial('medin', row['experiment'], row['participant'], source_row,
                                    index, phase, kind, item_index, span.start(), prefix, keys,
                                    label, correct, question, stimulus.group(1), feedback))
        counts[(phase, kind)] += 1
    if counts != Counter({('learning', 'category_choice'): 150,
                          ('test', 'category_choice'): 8, ('test', 'typicality_rating'): 8}):
        raise ValueError(f'Unexpected complete Medin source schema: {counts}')
    return trials


def parse_weather(row, source_row):
    text, spans, initial = _responses(row)
    m = re.search(r'rainy weather \(by pressing ([A-Z])\) or fine weather \(by pressing ([A-Z])\)', initial)
    if m is None or m.group(1) == m.group(2):
        raise ValueError('Weather keys must come from the original visible instructions.')
    keys = m.groups()
    outcome_keys = {'rainy': keys[0], 'fine': keys[1]}
    question = {'type': 'choice', 'instructions': 'Which weather do you predict for the current combination of cards?',
                'criteria': {keys[0]: 'Predict rainy weather.', keys[1]: 'Predict fine weather.'}}
    trials = []
    for index, span in enumerate(spans):
        prefix, label = text[:span.start()], span.group(1)
        stimulus = re.search(r'You are seeing the following: (card [1-4](?:, card [1-4]){0,2})\. You press $', prefix)
        feedback = re.match(r'\. You are (correct|wrong), the weather is (?:indeed )?(rainy|fine)\.', text[span.end():])
        if stimulus is None or feedback is None or label not in keys:
            raise ValueError('Unparsed original weather stimulus, response or feedback.')
        correct = outcome_keys[feedback.group(2)]
        if (label == correct) != (feedback.group(1) == 'correct'):
            raise ValueError('Original weather feedback is inconsistent with its stated keys.')
        trials.append(CategoryTrial('weather', row['experiment'], row['participant'], source_row,
            index, 'learning', 'weather_choice', index, span.start(), prefix, keys, label,
            correct, question, stimulus.group(1), feedback.group(0)))
    if len(trials) != 200:
        raise ValueError('Expected the complete original 200-trial weather session.')
    return trials


def load_category_trials(task, source=DEFAULT_SOURCE):
    parser = {'medin': parse_medin, 'weather': parse_weather}[task]
    trials = []
    with Path(source).open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] in TASK_SOURCES[task]:
                trials.extend(parser(row, index))
    if not trials:
        raise ValueError('No matching sessions in the supplied official test source.')
    return trials


def inventory(trials):
    sessions = {}
    for t in trials:
        s = sessions.setdefault(t.session_id, {'experiment': t.experiment, 'participant': t.participant,
            'source_row': t.source_row, 'responses': 0, 'response_kinds': {}})
        s['responses'] += 1
        kind = t.phase + '/' + t.response_kind
        s['response_kinds'][kind] = s['response_kinds'].get(kind, 0) + 1
    return {'task': trials[0].task, 'protocol_version': PROTOCOL_VERSION,
        'sessions': len(sessions), 'responses': len(trials), 'session_counts': sessions,
        'unique_experiment_participant_pairs': len({(t.experiment, str(t.participant)) for t in trials}),
        'response_kinds': dict(Counter(t.phase + '/' + t.response_kind for t in trials)),
        'history': 'Exact original text prefix before current response marker, including all past human responses and feedback; no truncation.',
        'split': 'Original Psych-101-test participant holdout; no resplitting.',
        'rating_note': 'Medin 1–9 typicality is an ordinal human rating, not success probability; typed-choice distribution is an added readout.',
        'objective_note': 'Medin learning feedback and realized weather outcomes are scored only; no ground truth invented for Medin test choices/ratings.'}


def score(trial, prediction):
    row = score_prediction(trial, prediction)
    if trial.correct_category is None:
        row['category_correct_under_human_history'] = None
    row['native_rating_absolute_error'] = (
        abs(int(row['choice']) - int(trial.original_choice_label))
        if trial.response_kind == 'typicality_rating' and not row['invalid_response'] else None)
    row['objective_target_semantics'] = ('Realized stochastic weather outcome; not the most probable weather.'
        if trial.task == 'weather' else 'Original learning correctness; unavailable for test category/rating fields.')
    return row


def _summarize_group(rows):
    zeros = sum(r['typed_choice_nll_is_infinite'] for r in rows)
    objective = [r['category_correct_under_human_history'] for r in rows
                 if r['category_correct_under_human_history'] is not None]
    ratings = [r['native_rating_absolute_error'] for r in rows if r['native_rating_absolute_error'] is not None]
    return {'responses': len(rows), 'typed_choice_mean_nll': None if zeros else statistics.mean(r['typed_choice_nll'] for r in rows),
        'typed_choice_mean_nll_is_infinite': bool(zeros), 'zero_human_probability_count': zeros,
        'human_response_agreement': statistics.mean(r['matches_human_choice'] for r in rows),
        'objective_label_accuracy_under_human_history': statistics.mean(objective) if objective else None,
        'objective_scored_responses': len(objective),
        'native_rating_MAE': statistics.mean(ratings) if ratings else None,
        'rating_scored_responses': len(ratings),
        'invalid_response_rate': statistics.mean(r['invalid_response'] for r in rows)}


def summarize_category(rows, expected):
    groups = {}
    for row in rows:
        key = row['experiment'] + ':' + row['phase'] + '/' + row['response_kind']
        groups.setdefault(key, []).append(row)
    result = {'status': 'complete' if len(rows) == expected else 'partial',
        'responses': len(rows), 'expected_responses': expected,
        'by_experiment_phase_response_kind': {k: _summarize_group(v) for k, v in groups.items()},
        'original_protocol_nll': None, 'original_protocol_nll_status': 'not_run_by_this_typed_runner',
        'interpretation': 'Human-history prediction; no autonomous closed-loop reward.',
        'timing': 'Shared-device batch/scalar wall time, not human reaction time.'}
    if rows:
        result.update(_summarize_group(rows))
    return result
