"""Exact-source response inputs for the remaining numeric and self-assessment tasks.

No model is loaded here. Human histories are exact source prefixes; answer keys,
CSV scores and stimulus difficulty are scoring-only metadata. Grammar and logic
retain different source splits. Population stimuli have no invented human IDs.
"""
from collections import Counter
import csv
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import re

from src.psych_category_tasks import CategoryTrial
from src.psych_typed import DEFAULT_SOURCE, RESPONSE, score_prediction

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'literature/four_task_resolution_20261018'
LOGIC_SOURCE = SOURCES / 'generalization__jansen2021logic__prompts.jsonl'
ASSESSMENT_ORDER = ('absAssess0', 'relAssess0', 'diffOther0', 'diffSelf0',
                    *(f'Q{i}' for i in range(1, 21)),
                    'absAssess1', 'relAssess1', 'diffOther1', 'diffSelf1')
PROTOCOL_VERSION = 'full-original-prefix-integer-and-self-assessment-v1'


def normalized_text(value):
    # Original Qualtrics uses curly apostrophes; Centaur's literal strings use
    # ASCII apostrophes. This only matches scoring keys, never changes input text.
    value = html.unescape(re.sub('<[^>]+>', '', value)).replace('’', "'").replace('‘', "'")
    return re.sub(r'\s+', ' ', value).strip()


def survey_metadata(domain):
    filename = ('Grammar' if domain == 'grammar' else 'LSAT') + '_SA_MTurk_-_attchecks.qsf'
    survey = json.loads((SOURCES / filename).read_text())
    correct, scales = {}, {}
    for element in survey['SurveyElements']:
        p = element.get('Payload')
        if not isinstance(p, dict):
            continue
        tag = p.get('DataExportTag', '')
        graded = [g for g in p.get('GradingData', [])
                  if isinstance(g, dict) and any(float(v) == 1 for v in g.get('Grades', {}).values())]
        if re.fullmatch(r'Q(?:[1-9]|1[0-9]|20)', tag) and graded:
            assert len(graded) == 1, (tag, graded)
            correct[tag] = normalized_text(p['Choices'][str(graded[0]['ChoiceID'])]['Display'])
        if tag in ASSESSMENT_ORDER and not tag.startswith('Q'):
            if tag.startswith('absAssess'):
                scales[tag] = [int(v['Display']) for v in p['Choices'].values()]
            elif tag.startswith('relAssess'):
                cfg = p['Configuration']
                assert (cfg['CSSliderMin'], cfg['CSSliderMax'], cfg['NumDecimals']) == (0, 100, '0')
                scales[tag] = list(range(101))
            else:
                scales[tag] = [int(p['Answers'][str(k)]['Display']) for k in p['AnswerOrder']]
    assert len(correct) == 20 and len(scales) == 8, (domain, correct.keys(), scales.keys())
    return {'source': str(SOURCES / filename), 'correct_answers': correct, 'original_scales': scales}


def raw_assessment_sessions(domain):
    filename = ('openloop__jansen2021dunningkruger__exp1.csv' if domain == 'grammar'
                else 'generalization__jansen2021logic__exp1.csv')
    result = {}
    with (SOURCES / filename).open() as stream:
        for row in csv.DictReader(stream):
            participant = str(row['participant'])
            target = result.setdefault(participant, {})
            assert row['question'] not in target
            target[row['question']] = row
    for participant, rows in result.items():
        assert set(rows) == set(ASSESSMENT_ORDER) | {'score'}, participant
    return result


def iter_assessment_sessions(domain):
    source = DEFAULT_SOURCE if domain == 'grammar' else LOGIC_SOURCE
    experiment = 'jansen2021dunningkruger/exp1.csv' if domain == 'grammar' else 'jansen2021logic/exp1.csv'
    raw, survey = raw_assessment_sessions(domain), survey_metadata(domain)
    with source.open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] != experiment:
                continue
            text, participant = row['text'], str(row['participant'])
            spans = list(RESPONSE.finditer(text))
            assert len(spans) == text.count('<<') == text.count('>>') == 28
            original = raw[participant]
            trials, human_correct = [], 0
            for index, (tag, span) in enumerate(zip(ASSESSMENT_ORDER, spans)):
                prefix, label = text[:span.start()], span.group(1)
                correct = None
                if tag.startswith('Q'):
                    stimulus = prefix[prefix.rfind('\n' + tag + '. ') + 1:]
                    option_text = stimulus.split('The choices are:\n', 1)[1].removesuffix('You press ')
                    options = dict(re.findall(r'^([A-Z]): (.*)$', option_text, re.M))
                    assert len(options) == 5 and label in options, (domain, participant, tag)
                    assert normalized_text(options[label]) == normalized_text(original[tag]['choice'])
                    targets = [key for key, value in options.items()
                               if normalized_text(value) == survey['correct_answers'][tag]]
                    assert len(targets) == 1, (domain, participant, tag, survey['correct_answers'][tag], options)
                    correct = targets[0]
                    human_correct += label == correct
                    keys = tuple(options)
                    instructions = 'Select the answer to the current question.'
                    phase, kind = 'questions', 'answer_choice'
                    item_index = int(tag[1:])
                else:
                    stimulus = prefix.split('\n')[-2]
                    assert label == original[tag]['choice'], (domain, participant, tag)
                    # Preserve the published benchmark transcript. The original QSF
                    # says 1..10 for difficulty; that discrepancy is reported separately.
                    maximum = 20 if tag.startswith('absAssess') else 100 if tag.startswith('relAssess') else 10
                    keys = tuple(map(str, range(maximum + 1)))
                    instructions = 'Integer count.' if tag.startswith('absAssess') else 'Integer percentile.' if tag.startswith('relAssess') else 'Difficulty rating.'
                    phase, kind = ('pre' if tag.endswith('0') else 'post'), tag[:-1]
                    item_index = index
                assert label in keys
                question = {'type': 'choice', 'instructions': instructions,
                            'criteria': {key: '' for key in keys}}
                trials.append(CategoryTrial('answer_self_assessment', experiment, row['participant'],
                    source_row, index, phase, kind, item_index, span.start(), prefix,
                    keys, label, correct, question, stimulus, None))
            assert human_correct == int(original['score']['choice']), (domain, participant, human_correct, original['score'])
            yield trials, {'source': str(source), 'source_row': source_row, 'participant': participant,
                           'domain': domain, 'responses': len(trials), 'raw_score': human_correct,
                           'raw_score_is_not_a_response': True,
                           'split': 'Psych-101-test participant holdout' if domain == 'grammar' else 'Centaur independent logic OOD; all published participants',
                           'all_response_labels_match_original_csv': True,
                           'recomputed_score_matches_original_csv': True}


def iter_probability_sessions():
    with DEFAULT_SOURCE.open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] not in ('zhu2020bayesian/exp1.csv', 'zhu2020bayesian/exp2.csv'):
                continue
            text = row['text']
            spans = list(RESPONSE.finditer(text))
            expected = 120 if row['experiment'].endswith('exp1.csv') else 180
            assert len(spans) == text.count('<<') == text.count('>>') == expected
            keys = tuple(map(str, range(101)))
            question = {'type': 'choice', 'instructions': 'Integer percentage.',
                        'criteria': {key: '' for key in keys}}
            trials = []
            for index, span in enumerate(spans):
                prefix, label = text[:span.start()], span.group(1)
                assert prefix.endswith('You estimate ') and label in keys
                query = prefix.split('\n')[-1].removesuffix(' You estimate ')
                assert text[span.end():span.end() + 2] == '%.'
                trials.append(CategoryTrial('probability_coherence_judgment', row['experiment'],
                    row['participant'], source_row, index, 'judgment', 'integer_percentage',
                    index, span.start(), prefix, keys, label, None, question, query, None))
            yield trials, {'source': str(DEFAULT_SOURCE), 'source_row': source_row,
                           'participant': row['participant'], 'responses': len(trials),
                           'split': 'Psych-101-test participant holdout'}


def score_response(trial, prediction):
    row = score_prediction(trial, prediction)
    row['category_correct_under_human_history'] = (None if trial.correct_category is None
                                                  else row['category_correct_under_human_history'])
    if trial.response_kind != 'answer_choice':
        row['reported_integer'] = int(row['choice']) if not row['invalid_response'] else None
        row['human_integer'] = int(trial.original_choice_label)
        row['native_scale_absolute_error'] = (abs(row['reported_integer'] - row['human_integer'])
                                              if row['reported_integer'] is not None else None)
        row['expected_numeric_report'] = sum(int(k) * p for k, p in row['probabilities'].items())
        row['numeric_unit'] = {'integer_percentage': 'percentage points of weather probability judgment',
                              'absAssess': 'number of correct answers, 0 to 20',
                              'relAssess': 'percentile rank, not event probability',
                              'diffOther': 'difficulty rating in source transcript, 0 to 10',
                              'diffSelf': 'difficulty rating in source transcript, 0 to 10'}[trial.response_kind]
    row['event_brier'] = None
    row['event_brier_status'] = 'No observed weather outcome or independent event-confidence probability supplied.'
    return row


def population_items():
    source = ROOT / 'runs/metacognition_stimuli_20261017/original_stimuli.csv'
    with source.open() as stream:
        for row in csv.DictReader(stream):
            row['id'] = f'benwell2022:form={row["form"]}:excel_row={row["source_excel_row"]}'
            yield row


def population_choice_request(item):
    return {'state': 'Which country has the larger population?\n'
            f'Left: {item["country_left"]}\nRight: {item["country_right"]}',
            'output_keys': ('Left', 'Right'),
            'question': {'type': 'choice', 'instructions': 'Choose the country with the larger population.',
                         'criteria': {'Left': '', 'Right': ''}}}


def population_confidence_request(item, choice):
    if choice not in ('Left', 'Right'):
        raise ValueError('Confidence must be conditioned on the actual preceding legal choice.')
    state = population_choice_request(item)['state'] + f'\nYou chose {choice}.'
    return {'state': state, 'output_keys': tuple(map(str, range(1, 7))),
            'question': {'type': 'choice', 'instructions': 'How confident are you in your choice? 1 = low confidence; 6 = high confidence.',
                         'criteria': {str(k): '' for k in range(1, 7)}}}


def score_population(item, choice, confidence):
    if choice not in ('Left', 'Right') or str(confidence) not in tuple(map(str, range(1, 7))):
        raise ValueError('Use a legal country choice and a separate original 1–6 confidence report.')
    difficulty = item['signed_difficulty_bin']
    if difficulty in ('', None):
        if item['phase'] != 'practice':
            raise ValueError(f'Formal population item is missing signed_difficulty_bin: {item["id"]}')
        difficulty = None
    else:
        difficulty = int(float(difficulty))
    return {'id': item['id'], 'form': item['form'], 'phase': item['phase'],
            'choice': choice, 'correct': choice == item['correct_side'], 'confidence_ordinal_1_to_6': int(confidence),
            'signed_difficulty_bin': difficulty,
            'human_participant_id': None, 'human_response_nll': None,
            'human_mapping_status': 'Original MAT files do not identify the country pair or form.',
            'event_brier': None, 'confidence_is_event_probability': False}
