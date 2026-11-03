"""Public CogGym social tasks and original ToMi test stories, with explicit typed adaptations."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics

from external.coggym_social_evaluation.experiment import Experiment, Trial, Query, Instruction
from external.coggym_social_evaluation.prompt_builder import _strip_html, _get_condition_instructions
from external.coggym_social_evaluation.scorer import score_multi_choice, score_single_slider
from external.coggym_social_evaluation.analyze import _collect_score, _normalize_group, _pearson_r2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'literature/task_catalog_20261017/raw'
TOMI = ROOT / 'data/tomi_original_2019/tomi_balanced_story_types'
VERSION = 'social-public-full-materials-typed-v1'
TASKS = {
    'politeness_pragmatics': [('Yoon2020Polite', 'exp1'), ('Yoon2020Polite', 'exp2')],
    'pragmatic_language_interpretation': [('Hu2023Fine', 'exp1')],
    'nonliteral_number_interpretation': [('Tsvilodub2025Nonliteral', 'exp1'), ('Tsvilodub2025Nonliteral', 'exp3b')],
    'false_belief_story_qa': [],
}


def source_path(study, exp, name):
    return RAW / f'coggym_repo__EML__{study}__{exp}__{name}'


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@dataclass
class SocialTrial:
    task: str
    experiment: str
    trial_id: str
    query_index: int
    query_tag: str
    state: str
    options: tuple
    kind: str
    original_prompt: str
    metadata: dict
    human_values: tuple = ()
    human_n: int = 0
    correct_answer: str | None = None

    @property
    def id(self):
        return f'{self.experiment}:{self.trial_id}:{self.query_tag or self.query_index}'

    @property
    def output_keys(self):
        return tuple(str(i + 1) for i in range(len(self.options)))

    @property
    def question(self):
        if self.kind == 'single-slider':
            # This is explicitly the probability of the stated event, not a choice's confidence.
            event = re.sub(r'^Please provide the probability as a percentage \(0-100%\) that ', '', self.original_prompt)
            return {'type': 'noul', 'instructions': event.rstrip('.?') + '.'}
        return {'type': 'choice', 'instructions': 'Choose your answer to the TARGET QUESTION in the supplied text.',
                'criteria': {key: f'Option {key} in the TARGET QUESTION.' for key in self.output_keys}}

    def identifiers(self):
        return {'id': self.id, 'task': self.task, 'experiment': self.experiment,
                'trial_id': self.trial_id, 'query_index': self.query_index, 'query_tag': self.query_tag,
                'kind': self.kind, 'output_keys': list(self.output_keys), 'options': list(self.options),
                'original_prompt': self.original_prompt, **self.metadata,
                'human_values_scoring_only': list(self.human_values), 'human_response_count': self.human_n,
                'correct_answer_scoring_only': self.correct_answer}


def load_coggym(task):
    selection = json.loads((RAW / 'coggym_repo__evaluation__trial_selection_map.json').read_text())
    output = []
    for study, exp in TASKS[task]:
        name = f'{study}/{exp}'
        config = json.loads(source_path(study, exp, 'config.json').read_text())
        instructions = [Instruction.from_dict(x) for x in read_lines(source_path(study, exp, 'instruction.jsonl'))]
        rows = read_lines(source_path(study, exp, 'trial.jsonl'))
        human = json.loads(source_path(study, exp, 'human_data_mean.json').read_text())
        individuals = json.loads(source_path(study, exp, 'human_data_ind.json').read_text())
        # Use the source class for its exact condition-to-instruction mapping.
        experiment = object.__new__(Experiment)
        experiment.config, experiment.instructions = config, instructions
        experiment.path = Path(name)
        selected = set(selection[name]['selected_ids'])
        practices = {x.stimuli_id for x in instructions if x.type == 'test_trial'}
        for row in rows:
            tid = row['id']
            if tid in practices:
                continue  # Source-defined checks; all remain in original source and inventory.
            visible_instructions = _get_condition_instructions(experiment, tid)
            parts = [_strip_html(x.text) for x in visible_instructions]
            for stimulus in row['stimuli']:
                if stimulus['input_type'] != 'text':
                    raise ValueError('This entry implements the selected public text-only studies.')
                parts.append(_strip_html(stimulus['text']))
            queries = row['queries']
            # Preserve all original co-present questions/options, not earlier invented answers.
            for q in queries:
                parts.append(_strip_html(q['prompt']))
                if q['type'] == 'multi-choice':
                    parts.extend(f'{i + 1}. {_strip_html(opt)}' for i, opt in enumerate(q['option']))
                elif q['type'] == 'single-slider':
                    parts.append('Original slider: ' + json.dumps(q['slider_config'], ensure_ascii=False))
            for qi, query in enumerate(queries):
                if query['type'] == 'text-instruction':
                    continue
                kind, tag = query['type'], query['tag']
                if kind not in {'multi-choice', 'single-slider'}:
                    raise ValueError(f'Unsupported original query type {kind}')
                options = tuple(_strip_html(x) for x in query.get('option', []))
                hraw = human[tid][tag]
                individual = individuals[tid][tag]
                if kind == 'multi-choice':
                    values = tuple(float(hraw[f'{tag}_{i + 1}']) for i in range(len(options)))
                    counts = Counter(x['idx'] for x in individual)
                    for i, opt in enumerate(options):
                        if any(x['option_text'] != query['option'][i] for x in individual if x['idx'] == i):
                            raise ValueError('Human source index and original option text disagree.')
                        if not math.isclose(values[i] / sum(values), counts[i] / len(individual), abs_tol=1e-12):
                            raise ValueError('Human individual responses disagree with released option means/counts.')
                else:
                    values = (float(hraw),)
                prompt = _strip_html(query['prompt'])
                target = ['TARGET QUESTION:', prompt]
                target.extend(f'{i + 1}. {opt}' for i, opt in enumerate(options))
                condition = tid.split('_')[0] if study == 'Hu2023Fine' else row.get('condition', '')
                if study == 'Yoon2020Polite' and exp == 'exp2':
                    match = re.fullmatch(r'trial_(both|social|informative)_([0-3])_c([1-4])', tid)
                    if match is None:
                        raise ValueError('Unexpected original Yoon goal/state/layout ID.')
                    condition = ':'.join(match.groups())
                elif study == 'Yoon2020Polite':
                    hearts = re.search(r'\(([0-3]) out of 3 hearts\)', '\n'.join(parts))
                    condition = 'literal:state=' + hearts.group(1)
                elif study == 'Tsvilodub2025Nonliteral':
                    condition = exp + ':' + ('electric_kettle' if 'electric kettle' in parts[2] else 'watch' if 'watch' in parts[2] else 'laptop')
                metadata = {'official_selected': tid in selected, 'condition': condition,
                            'source_metadata': row.get('metadata'), 'source_query': query,
                            'instructions_used': [x.id for x in visible_instructions],
                            'source_url': f'https://github.com/lance-ying/coggym/tree/main/EML/{name}',
                            'protocol': 'Public CogGym reconstruction; added per-query typed interface; no claim of original-human session replay.',
                            'joint_response_note': 'Co-present questions remain visible. Per-query marginal calls do not assert a joint sentence distribution.',
                            'human_weights_are_counts': kind == 'multi-choice' and sum(values) > 1.001}
                answer = query.get('answer')
                output.append(SocialTrial(task, name, tid, qi, tag, '\n\n'.join(parts + target),
                    options, kind, prompt, metadata, values, len(individual),
                    options[answer] if isinstance(answer, int) and kind == 'multi-choice' else None))
        if not selected.issubset({t.trial_id for t in output if t.experiment == name}):
            raise ValueError('Every official selected trial must be retained.')
    return output


def load_tomi(split='test'):
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Use an original archive split.')
    traces = (TOMI / f'fb_all_{split}.trace').read_text().splitlines()
    trials, story = [], []
    for source_line, line in enumerate((TOMI / f'fb_all_{split}.txt').read_text().splitlines()):
        number, sentence = line.split(' ', 1)
        if number == '1':
            story = []
        if '\t' not in sentence:
            story.append(sentence)
            continue
        prompt, answer, supporting = sentence.split('\t')
        trace = traces[len(trials)].split(',')
        # Derive only from visible object-location events; never from labels, support IDs or trace.
        containers = []
        for event in story:
            match = re.search(r'(?:is in|moved .+ to) the (\w+)\.$', event)
            if match and match.group(1) not in containers:
                containers.append(match.group(1))
        if answer not in containers or len(containers) < 2:
            raise ValueError('Source answer must be in the answer-independent visible container set.')
        state = '\n'.join(story) + '\n\nTARGET QUESTION:\n' + prompt + '\n'
        state += '\n'.join(f'{i + 1}. {x}' for i, x in enumerate(containers))
        metadata = {'official_selected': True, 'split': split, 'source_line': source_line,
            'original_story_block': len(trials) // 6,
            'story_type': trace[-1], 'question_type': trace[-2], 'source_trace_scoring_only': traces[len(trials)],
            'supporting_fact_indices_scoring_only': supporting,
            'condition': trace[-1] + ':' + trace[-2], 'human_data': False,
            'protocol': 'Original full stories and split; added visible-container multiple-choice restriction. Exact original answer string scoring; not original unrestricted QA.',
            'candidate_rule': 'All containers in visible object placement/movement statements, first-mention order; no labels used.',
            'source_url': 'https://github.com/facebookresearch/ToMi'}
        trials.append(SocialTrial('false_belief_story_qa', f'ToMi/{split}', str(len(trials)), 0,
            'answer', state, tuple(containers), 'multi-choice', prompt, metadata,
            correct_answer=answer))
    if len(traces) != len(trials):
        raise ValueError('Original trace and question counts differ.')
    for begin in range(0, len(trials), 6):
        group = trials[begin:begin + 6]
        if len(group) != 6 or len({x.state.split('\n\nTARGET QUESTION:')[0] for x in group}) != 1:
            raise ValueError('Original contiguous six-question story block not preserved.')
    return trials


def load_trials(task):
    return load_tomi() if task == 'false_belief_story_qa' else load_coggym(task)


def inventory(trials):
    tomi = trials[0].task == 'false_belief_story_qa'
    return {'protocol_version': VERSION, 'task': trials[0].task, 'responses': len(trials),
        'trials': len({(x.experiment, x.trial_id) for x in trials}),
        'experiment_responses': dict(Counter(x.experiment for x in trials)),
        'official_selected_responses': sum(t.metadata['official_selected'] for t in trials),
        'condition_responses': dict(Counter(t.metadata['condition'] for t in trials)),
        'history': ('Complete original story for each question; no previous questions, answers, hidden traces or supporting-fact labels in input.' if tomi else 'Independent complete public scenarios; all co-present queries retained, no other scenario responses supplied.'),
        'split': ('Original ToMi archive test; train and validation retained unchanged. No resplit or training on test.' if tomi else 'CogGym published selection flagged separately within all non-practice public materials. Public scenarios have no official ML split and are not claimed unseen.'),
        'human_reference': ('None: ToMi stories and answer labels are program generated; no human participants or responses.' if tomi else 'Released aggregated responses plus original per-item individual arrays; no invented participant trajectories.'),
        'maximum_state_characters': max(len(x.state) for x in trials),
        'probability_semantics': 'Choice mass models a response; slider reports are explicit subjective event probabilities, with no observed event outcomes for Brier.'}


def score(trial, prediction):
    row = {**trial.identifiers(), **prediction, 'invalid_response': False}
    if trial.kind == 'single-slider':
        p = prediction['reported_probability']
        if p is None:
            row['invalid_response'] = True
            return row
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid numerical event report escaped the parser.')
        reported = 100 * p
        row.update(reported_percentage=reported,
                   absolute_error_to_human_mean=abs(reported - trial.human_values[0]))
        q = Query.from_dict(trial.metadata['source_query'])
        row['official_coggym_score'] = score_single_slider({'value': reported},
            Trial(trial.trial_id, [], [q]), q, {trial.trial_id: {trial.query_tag: trial.human_values[0]}})
        return row
    ps = prediction['probabilities']
    if len(ps) != len(trial.options) or any(not math.isfinite(x) or x < 0 for x in ps) or not math.isclose(sum(ps), 1, abs_tol=1e-6):
        raise ValueError('Invalid choice distribution.')
    argmax = max(range(len(ps)), key=lambda i: ps[i])
    native = prediction.get('native_choice')
    if native is not None and native not in trial.output_keys:
        row['invalid_response'] = True
    actual = (trial.output_keys.index(native) if native in trial.output_keys else None) if native is not None else argmax
    row.update(argmax_index=argmax, argmax_answer=trial.options[argmax],
               native_answer=trial.options[trial.output_keys.index(native)] if native in trial.output_keys else None,
               actual_choice_index=actual, actual_answer=trial.options[actual] if actual is not None else None,
               choice_source='native' if native is not None else 'argmax_no_native_choice')
    if trial.correct_answer is not None:
        row['original_answer_exact_match'] = actual is not None and trial.options[actual] == trial.correct_answer
        row['argmax_original_answer_exact_match_diagnostic'] = trial.options[argmax] == trial.correct_answer
    if trial.human_values:
        h = [x / sum(trial.human_values) for x in trial.human_values]
        zero = any(w > 0 and p == 0 for w, p in zip(h, ps))
        row['human_response_nll_is_infinite'] = zero
        row['human_response_mean_nll'] = None if zero else -sum(w * math.log(p) for w, p in zip(h, ps) if w)
        row['human_probability_of_argmax'] = h[argmax]
        row['human_probability_of_actual_choice'] = h[actual] if actual is not None else None
        row['human_mode_match'] = actual == max(range(len(h)), key=lambda i: h[i])
        row['argmax_human_mode_match_diagnostic'] = argmax == max(range(len(h)), key=lambda i: h[i])
        q = Query.from_dict(trial.metadata['source_query'])
        human = {trial.trial_id: {trial.query_tag: {f'{trial.query_tag}_{i + 1}': x for i, x in enumerate(trial.human_values)}}}
        row['official_coggym_score'] = score_multi_choice({'selected_index': actual if actual is not None else -1,
            'selected': trial.options[actual] if actual is not None else ''}, Trial(trial.trial_id, [], [q]), q, human)
        row['argmax_official_coggym_score_diagnostic'] = score_multi_choice({'selected_index': argmax,
            'selected': trial.options[argmax]}, Trial(trial.trial_id, [], [q]), q, human)
        row['official_raw_weight_note'] = 'Official scorer returns raw human counts for Hu; normalized probability is separately reported above.'
    return row


def summarize(rows, expected):
    def group_summary(group):
        result = {'responses': len(group), 'invalid_responses': sum(r['invalid_response'] for r in group)}
        result['invalid_response_rate'] = result['invalid_responses'] / len(group) if group else None
        times = sorted(r['batch_elapsed_seconds'] for r in group if r.get('batch_elapsed_seconds') is not None)
        for percentile in (50, 95):
            if times:
                index = (len(times) - 1) * percentile / 100
                lower, upper = math.floor(index), math.ceil(index)
                value = times[lower] + (index - lower) * (times[upper] - times[lower])
            else:
                value = None
            result[f'latency_p{percentile}_seconds'] = value
        for field in ['original_answer_exact_match', 'argmax_original_answer_exact_match_diagnostic',
                      'human_mode_match', 'human_probability_of_actual_choice',
                      'human_probability_of_argmax', 'absolute_error_to_human_mean']:
            values = [r[field] for r in group if r.get(field) is not None]
            if values:
                result[field + '_mean'] = statistics.mean(values)
        nll = [r for r in group if 'human_response_mean_nll' in r]
        if nll:
            result['human_response_mean_nll_is_infinite'] = any(r['human_response_nll_is_infinite'] for r in nll)
            result['human_response_weighted_mean_nll'] = (None if result['human_response_mean_nll_is_infinite'] else sum(r['human_response_mean_nll'] * r['human_response_count'] for r in nll) / sum(r['human_response_count'] for r in nll))
        return result
    result = {'status': 'complete' if len(rows) == expected else 'partial', 'expected_responses': expected,
            **group_summary(rows), 'official_selected_subset': group_summary([r for r in rows if r['official_selected']]),
            'by_experiment': {e: group_summary([r for r in rows if r['experiment'] == e]) for e in sorted({r['experiment'] for r in rows})},
            'by_condition': {e: group_summary([r for r in rows if r['condition'] == e]) for e in sorted({r['condition'] for r in rows})},
            'timing': 'Shared-device wall latency, not human reaction time.',
            'interpretation': 'Added typed public-scenario evaluation; no closed-loop return or original participant-history prediction.'}
    if rows and rows[0]['task'] == 'false_belief_story_qa':
        groups = {}
        for row in rows:
            groups.setdefault(row['original_story_block'], []).append(row)
        full = [group for group in groups.values() if len(group) == 6]
        result['original_story_joint_accuracy'] = (statistics.mean(all(r['original_answer_exact_match'] for r in g) for g in full) if full else None)
        result['complete_six_question_story_blocks'] = len(full)
    if rows and rows[0]['task'] != 'false_belief_story_qa':
        metrics = {}
        for selected_only in (True, False):
            experiments = {}
            for experiment in sorted({r['experiment'] for r in rows}):
                groups = defaultdict(lambda: {'bounds': [None, None], 'items': defaultdict(lambda: {'model': [], 'human': []})})
                for row in rows:
                    if row['experiment'] != experiment or (selected_only and not row['official_selected']):
                        continue
                    source_query = row['source_query']
                    bounds = source_query.get('slider_config', {})
                    qmeta = {'type': row['kind'], 'scale_min': bounds.get('min'),
                             'scale_max': bounds.get('max'), 'n_options': len(row['options'])}
                    official = row.get('official_coggym_score', {})
                    if official.get('scorable'):
                        _collect_score(groups, row['trial_id'], row['query_tag'], official, qmeta)
                xs, ys = [], []
                for group in groups.values():
                    x, y = _normalize_group(group)
                    xs.extend(x)
                    ys.extend(y)
                experiments[experiment] = {'pearson_r2': _pearson_r2(xs, ys) if len(xs) >= 5 else None,
                    'scored_option_or_slider_items': len(xs), 'source': 'Unmodified public CogGym analyze.py helpers; one frozen response per item, no repeated-model average.'}
            metrics['official_selected' if selected_only else 'all_public_nonpractice'] = experiments
        result['coggym_original_aggregation'] = metrics
    return result
