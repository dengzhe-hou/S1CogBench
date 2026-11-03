"""Exact Psych-101 held-out risk descriptions and two-stage information search.

These are human-history prediction protocols. Every original response is kept;
the second search decision conditions on the human's immediately preceding key.
No unobserved payoff, model-generated reward, or success probability is invented.
"""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from src.psych_category_tasks import CategoryTrial, summarize_category
from src.psych_typed import DEFAULT_SOURCE, RESPONSE, score_prediction

TASK_SOURCES = {
    'choices13k': 'peterson2021using/exp1.csv',
    'risky_choice_information_search': 'krueger2022identifying/exp1.csv',
}
PROTOCOL_VERSION = 'psych101-risk-full-prefix-typed-v1'
NUMBER = r'-?\d+(?:\.\d+)?'
OPTION = re.compile(r'^Option ([A-Z]) delivers (.+)\.$', re.M)
OUTCOME = re.compile(r'(' + NUMBER + r') points with (unknown|\d+(?:\.\d+)?%) chance')
CHOICE_FEEDBACK = re.compile(r'\. You receive (' + NUMBER + r') points by selecting this option\. '
                             r'You would have received (' + NUMBER + r') points had you chosen the other option\.')
SEARCH_ROUND = re.compile(r'A new round begins\.\nThere are (.+)\.\n')


def source_spans(row, expected):
    if row['experiment'] != expected:
        raise ValueError('Task source must match the named original paper.')
    text = row['text']
    spans = list(RESPONSE.finditer(text))
    if not spans or len(spans) != text.count('<<') or len(spans) != text.count('>>'):
        raise ValueError('Every original response marker must parse exactly once.')
    return text, spans


def parse_choices13k(row, source_row):
    text, spans = source_spans(row, TASK_SOURCES['choices13k'])
    options = list(OPTION.finditer(text))
    if not options or len(options) % 2:
        raise ValueError('Every original lottery problem needs its two complete descriptions.')
    trials, cursor = [], 0
    for pi in range(len(options) // 2):
        pair = options[2*pi:2*pi+2]
        keys = tuple(m.group(1) for m in pair)
        if len(set(keys)) != 2:
            raise ValueError('The original two lottery keys must be distinct.')
        end = options[2*pi+2].start() if 2*pi+2 < len(options) else len(text)
        descriptions = {m.group(1): m.group(2) for m in pair}
        lotteries = {}
        for key, description in descriptions.items():
            outcomes = list(OUTCOME.finditer(description))
            residue = OUTCOME.sub('', description)
            if not outcomes or re.sub(r'either|or|,|\s', '', residue):
                raise ValueError('An original outcome/probability description is unparsed.')
            values = [{'points': float(m.group(1)), 'probability': None if m.group(2) == 'unknown'
                       else float(m.group(2)[:-1])/100} for m in outcomes]
            known = all(x['probability'] is not None for x in values)
            lotteries[key] = {'outcomes': values, 'all_probabilities_known': known,
                'stated_probability_sum': sum(x['probability'] for x in values) if known else None,
                'expected_points_from_explicit_description': sum(x['points']*x['probability'] for x in values) if known else None}
        question = {'type': 'choice', 'instructions': 'Which of the two current gambling options do you select?',
                    'criteria': {key: f'Select option {key}.' for key in keys}}
        within = 0
        while cursor < len(spans) and spans[cursor].start() < end:
            span = spans[cursor]
            if span.start() < pair[1].end() or not text[:span.start()].endswith('You press ') or span.group(1) not in keys:
                raise ValueError('Choice must follow both current lottery descriptions and use their original keys.')
            suffix = text[span.end():end].split('\n', 1)[0]
            feedback = CHOICE_FEEDBACK.fullmatch(suffix)
            if suffix != '.' and feedback is None:
                raise ValueError('Original shown/suppressed feedback must parse without modification.')
            observed = {'raw': suffix, 'feedback_shown': feedback is not None,
                        'human_received_points': float(feedback.group(1)) if feedback else None,
                        'human_foregone_points': float(feedback.group(2)) if feedback else None}
            stimulus = {'problem_index': pi, 'response_in_problem': within,
                        'lotteries': lotteries, 'descriptions': descriptions}
            trials.append(CategoryTrial('choices13k', row['experiment'], row['participant'], source_row,
                cursor, 'original_session', 'lottery_choice', pi, span.start(), text[:span.start()], keys,
                span.group(1), None, question, json.dumps(stimulus), observed))
            cursor, within = cursor+1, within+1
        if within != 5:
            raise ValueError('The original Peterson problem must retain all five repeated choices.')
    if cursor != len(spans):
        raise ValueError('Unassigned original Peterson response; no row may be skipped.')
    return trials


def parse_information_search(row, source_row):
    text, spans = source_spans(row, TASK_SOURCES['risky_choice_information_search'])
    initial = text[:spans[0].start()]
    mapping = re.search(r'6 different gambles labeled: (.+)\.', initial)
    cost = re.search(r'Each time you check a gamble will cost you (\d+) points\.', initial)
    if mapping is None or cost is None:
        raise ValueError('Original gamble labels and inspection cost must be visible.')
    keys = tuple(re.findall(r'\b[A-Z]\b', mapping.group(1)))
    if len(keys) != 6 or len(set(keys)) != 6:
        raise ValueError('Exactly six distinct original gamble keys are required.')
    rounds = list(SEARCH_ROUND.finditer(text))
    if len(rounds) != 20:
        raise ValueError('All original twenty search rounds must be retained.')
    trials, cursor = [], 0
    for ri, rm in enumerate(rounds):
        end = rounds[ri+1].start() if ri+1 < len(rounds) else len(text)
        colors = re.findall(r'(\d+) ([a-z]+) balls', rm.group(1))
        if len(colors) != 4 or len({c for _,c in colors}) != 4:
            raise ValueError('The original four ball colors/counts must parse.')
        color_keys = tuple(c for _,c in colors)
        second_keys = color_keys + ('stop',)
        query = {'type':'choice', 'instructions':'Which gamble do you select for your next check or final choice?',
                 'criteria':{k:f'Choose gamble {k} for the next response.' for k in keys}}
        second_query = {'type':'choice', 'instructions':'For the gamble you just selected, choose the ball color to check or type stop to choose that gamble.',
                        'criteria':{**{c:f'Pay {cost.group(1)} points to reveal this gamble\'s payoff for a {c} ball.' for c in color_keys},
                                    'stop':'Stop checking and choose the just-selected gamble.'}}
        checks, step, ended = 0, 0, False
        while cursor < len(spans) and spans[cursor].start() < end:
            first, second = spans[cursor:cursor+2]
            if ended or first.start() < rm.end() or second.start() >= end or not text[:first.start()].endswith('You press '):
                raise ValueError('Two ordered responses must occur before each original round terminates.')
            if first.group(1) not in keys or second.group(1) not in second_keys or text[first.end():second.start()] != ' and then type ':
                raise ValueError('Original gamble/color/stop responses and ordering must be retained.')
            suffix = text[second.end():end].split('\n',1)[0]
            if second.group(1) == 'stop':
                feedback = re.fullmatch(r'\. A ([a-z]+) ball is chosen, and you earn (-?\d+) points\.', suffix)
                if feedback is None or feedback.group(1) not in color_keys:
                    raise ValueError('Original terminal payoff and sampled color must parse.')
                observed = {'human_action':'stop', 'sampled_color':feedback.group(1),
                            'human_observed_payoff':int(feedback.group(2)), 'raw':suffix}
                ended = True
            else:
                feedback = re.fullmatch(r'\. The payoff for this combination would be (-?\d+) points\.', suffix)
                if feedback is None:
                    raise ValueError('Every original information acquisition needs its revealed payoff.')
                observed = {'human_action':'inspect', 'human_revealed_payoff':int(feedback.group(1)), 'raw':suffix}
            visible = {'round_index':ri, 'step_in_round':step, 'inspection_cost':int(cost.group(1)),
                       'checks_before':checks, 'stated_ball_counts':dict((c,int(n)) for n,c in colors),
                       'stated_ball_total':sum(int(n) for n,_ in colors)}
            # The first input does not reveal the following human gamble choice.
            trials.append(CategoryTrial('risky_choice_information_search',row['experiment'],row['participant'],source_row,
                cursor,'original_session','gamble_selection',ri,first.start(),text[:first.start()],keys,first.group(1),None,
                query,json.dumps(visible),{'next_human_stage':second.group(1), **observed}))
            trials.append(CategoryTrial('risky_choice_information_search',row['experiment'],row['participant'],source_row,
                cursor+1,'original_session','inspect_or_stop',ri,second.start(),text[:second.start()],second_keys,second.group(1),None,
                second_query,json.dumps({**visible,'human_selected_gamble':first.group(1)}),observed))
            checks += not ended
            step, cursor = step+1, cursor+2
        if not ended:
            raise ValueError('Original round has no stop response; do not silently omit it.')
    if cursor != len(spans):
        raise ValueError('Every original search response must be retained.')
    return trials


def load_trials(task, source=DEFAULT_SOURCE):
    parser = {'choices13k':parse_choices13k, 'risky_choice_information_search':parse_information_search}[task]
    trials = []
    with Path(source).open() as stream:
        for source_row,line in enumerate(stream):
            row=json.loads(line)
            if row['experiment'] == TASK_SOURCES[task]:
                trials.extend(parser(row,source_row))
    if not trials:
        raise ValueError('No original task sessions found.')
    return trials


def inventory(trials):
    sessions={}
    for t in trials:
        s=sessions.setdefault(t.session_id,{'experiment':t.experiment,'participant':t.participant,
            'source_row':t.source_row,'responses':0,'problems':0})
        s['responses']+=1
        s['problems']=max(s['problems'],t.item_index+1)
    return {'task':trials[0].task,'protocol_version':PROTOCOL_VERSION,'sessions':len(sessions),'responses':len(trials),
        'problems':sum(s['problems'] for s in sessions.values()),'session_counts':sessions,
        'response_kinds':dict(Counter(t.response_kind for t in trials)),
        'maximum_full_prefix_characters':max(t.prefix_end for t in trials),
        'history':'Exact full original session prefix through current response cue; all prior human choices and feedback, no truncation or problem reset.',
        'split':'Original Psych-101-test participant holdout; no resplitting. Not the original Choices13k problem-level choice-fraction protocol.',
        'interpretation':'Human-history prediction; no model closed-loop payoff or objective action accuracy.',
        'original_protocol_nll_status':'not_run_by_this_typed_runner'}


def score(trial,prediction):
    row=score_prediction(trial,prediction)
    row.update(category_correct_under_human_history=None,native_rating_absolute_error=None,
               model_closed_loop_reward=None,objective_action_accuracy=None)
    if trial.response_kind == 'inspect_or_stop':
        row['model_stop_probability']=row['probabilities']['stop']
    return row


def summarize(rows,expected):
    result=summarize_category(rows,expected)
    times=sorted(r['batch_elapsed_seconds'] for r in rows)
    result.update(model_closed_loop_reward=None,objective_action_accuracy=None,
        latency_p50_seconds=statistics.median(times) if times else None,
        latency_p95_seconds=times[-(-95*len(times)//100)-1] if times else None)
    return result
