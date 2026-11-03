"""Typed-choice human-response prediction on original Psych-101 test transcripts.

Currently implements only Badham et al. SHJ categorization. Full prefixes, source
participant IDs and the published held-out split are retained. This is an added
typed-choice protocol, not the original full-vocabulary Centaur likelihood.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'data/psych101/prompts_testing_t1.jsonl'
EXPERIMENT = 'badham2017deficits/exp1.csv'
PROTOCOL_VERSION = 'psych101-shj-full-prefix-typed-v1'
RESPONSE = re.compile(r'<<([^<>]+)>>')
CATEGORIES = re.compile(r'whether an object belongs to the ([A-Z]) or ([A-Z]) category\.')
NEW_PROBLEM = 'You encounter a new problem with a new rule determining which objects belong to each category:'


@dataclass(frozen=True)
class ResponseTrial:
    experiment: str
    participant: object
    source_row: int
    trial_index: int
    problem_index: int
    problem_trial_index: int
    prefix_end: int
    state: str
    output_keys: tuple
    original_choice_label: str
    correct_category: str

    @property
    def session_id(self):
        # Source has no separate session column. Preserve that fact; the stable
        # JSONL source row identifies a source session without inventing an ID.
        return f'{self.experiment}:row={self.source_row}'

    @property
    def id(self):
        return f'{self.session_id}:response={self.trial_index}'

    @property
    def question(self):
        return {'type': 'choice', 'instructions': 'Which category do you choose for the current object?',
                'criteria': {key: f'Choose category {key}.' for key in self.output_keys}}

    def identifiers(self):
        return {'id': self.id, 'experiment': self.experiment, 'participant': self.participant,
                'session_id': self.session_id, 'session_id_origin': 'derived from original JSONL row',
                'source_row': self.source_row, 'trial_index': self.trial_index,
                'trial_id_origin': 'zero-based response position in the original session',
                'problem_index': self.problem_index, 'problem_trial_index': self.problem_trial_index,
                'prefix_end_character': self.prefix_end, 'output_keys': list(self.output_keys),
                'original_choice_label': self.original_choice_label,
                'correct_category_for_scoring_only': self.correct_category}


def parse_shj_session(row, source_row):
    """The current label and feedback are parsed for scoring, never into state."""
    if row['experiment'] != EXPERIMENT:
        raise ValueError('Only the explicitly implemented SHJ source is supported.')
    text = row['text']
    categories = CATEGORIES.search(text)
    if categories is None or categories.group(1) == categories.group(2):
        raise ValueError('Expected two distinct categories in the original SHJ instructions.')
    keys = categories.groups()
    responses = list(RESPONSE.finditer(text))
    if len(responses) != text.count('<<') or len(responses) != text.count('>>'):
        raise ValueError('Unparsed response marker in the source session.')
    problems = [m.start() for m in re.finditer(re.escape(NEW_PROBLEM), text)]
    if len(problems) != 4:
        raise ValueError('This SHJ source must retain all four original problems.')
    in_problem = Counter()
    result = []
    for trial_index, match in enumerate(responses):
        label = match.group(1)
        if label not in keys:
            raise ValueError('Original response is outside the categories stated to the participant.')
        prefix = text[:match.start()]
        if not prefix.endswith('You press '):
            raise ValueError('Unexpected SHJ response boundary.')
        feedback = re.match(r'\. The correct category is ([A-Z])\.', text[match.end():])
        if feedback is None or feedback.group(1) not in keys:
            raise ValueError('Missing or illegal original SHJ feedback.')
        problem_index = sum(p < match.start() for p in problems) - 1
        if problem_index < 0:
            raise ValueError('A response precedes the first original problem.')
        result.append(ResponseTrial(
            experiment=row['experiment'], participant=row['participant'], source_row=source_row,
            trial_index=trial_index, problem_index=problem_index,
            problem_trial_index=in_problem[problem_index], prefix_end=match.start(),
            state=prefix, output_keys=keys, original_choice_label=label,
            correct_category=feedback.group(1)))
        in_problem[problem_index] += 1
    return result


def load_shj_trials(source=DEFAULT_SOURCE):
    trials = []
    with Path(source).open() as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            if row['experiment'] == EXPERIMENT:
                trials.extend(parse_shj_session(row, source_row))
    if not trials:
        raise ValueError('No SHJ sessions found in the supplied official test file.')
    return trials


def source_inventory(trials):
    sessions = {}
    for trial in trials:
        session = sessions.setdefault(trial.session_id, {
            'source_row': trial.source_row, 'participant': trial.participant,
            'output_keys': list(trial.output_keys), 'responses': 0, 'problem_responses': {}})
        session['responses'] += 1
        problem = str(trial.problem_index)
        session['problem_responses'][problem] = session['problem_responses'].get(problem, 0) + 1
    return {'experiment': EXPERIMENT, 'sessions': len(sessions),
            'participants': len({json.dumps(t.participant) for t in trials}),
            'responses': len(trials), 'session_counts': sessions,
            'label_counts': dict(Counter(t.original_choice_label for t in trials)),
            'full_prefix_characters': {'minimum': min(len(t.state) for t in trials),
                                       'maximum': max(len(t.state) for t in trials)},
            'split': 'Original Psych-101-test participant holdout; no resplitting.',
            'history': 'Exact original text prefix, including past responses/feedback/markers; no truncation.',
            'scope': 'One task/source only; other 30 planned tasks are not completed by this runner.'}


def score_prediction(trial, prediction):
    """No probability smoothing in primary NLL; zero mass remains infinite."""
    probs = [float(v) for v in prediction['probabilities']]
    if (len(probs) != len(trial.output_keys) or
            any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or
            abs(sum(probs) - 1) > 1e-7):
        raise ValueError('Adapter must supply finite normalized legal-choice probabilities.')
    human_probability = probs[trial.output_keys.index(trial.original_choice_label)]
    native = prediction.get('native_choice')
    invalid = native is not None and native not in trial.output_keys
    argmax = trial.output_keys[max(range(len(probs)), key=probs.__getitem__)]
    choice = native if native is not None else argmax
    record = trial.identifiers()
    record.update({
        'probabilities': dict(zip(trial.output_keys, probs)),
        'native_choice': native, 'choice': choice,
        'choice_source': 'native_choice' if native is not None else 'legal_probability_argmax',
        'invalid_response': invalid, 'matches_human_choice': choice == trial.original_choice_label,
        'category_correct_under_human_history': choice == trial.correct_category,
        'human_choice_probability': human_probability,
        'typed_choice_nll': -math.log(human_probability) if human_probability else None,
        'typed_choice_nll_is_infinite': human_probability == 0,
        'typed_choice_nll_clipped_1e12': -math.log(max(human_probability, 1e-12)),
        'original_protocol_nll': None,
        'original_protocol_nll_status': 'Not scored; this is an added typed-choice prompt/normalization.',
    })
    for field in ['raw_probabilities', 'input_tokens', 'allowed_token_mass', 'label_mapping',
                  'batch_id', 'batch_size', 'batch_elapsed_seconds', 'inference_interface']:
        if field in prediction:
            record[field] = prediction[field]
    return record


def summarize(records, expected):
    count = len(records)
    zeros = sum(row['typed_choice_nll_is_infinite'] for row in records)
    def mean(field):
        return statistics.mean(row[field] for row in records) if count else None
    by_problem = {}
    for problem in range(4):
        group = [row for row in records if row['problem_index'] == problem]
        if group:
            by_problem[str(problem)] = {
                'responses': len(group),
                'category_accuracy_under_human_history': statistics.mean(
                    r['category_correct_under_human_history'] for r in group)}
    return {'status': 'complete' if count == expected else 'partial', 'responses': count,
            'expected_responses': expected,
            'typed_choice_mean_nll': None if zeros else mean('typed_choice_nll'),
            'typed_choice_mean_nll_is_infinite': bool(zeros), 'zero_human_probability_count': zeros,
            'typed_choice_mean_nll_clipped_1e12_diagnostic': mean('typed_choice_nll_clipped_1e12'),
            'human_choice_agreement': mean('matches_human_choice'),
            'category_accuracy_under_human_history': mean('category_correct_under_human_history'),
            'invalid_response_rate': mean('invalid_response'), 'problem_order_analysis': by_problem,
            'problem_order_note': 'Original presentation order, not inferred SHJ type I–VI labels.',
            'original_protocol_nll': None, 'original_protocol_nll_status': 'not_run_by_this_typed_runner',
            'interpretation': 'Human-history response prediction; no autonomous closed-loop reward.',
            'timing': 'Shared-device batch wall time. Batched throughput is not individual reaction time.'}


def audit_lengths(trials, models):
    """Tokenize every exact model input on CPU; no model weights or GPU work."""
    import sys
    from transformers import AutoTokenizer
    from src.hf_choice_models import FrozenDecoderChoice
    from examples.qwen35_4b_choice_agent import format_choice_prompt
    directories = {'llama8b': ROOT / 'models/llama-3.1-8b-instruct',
                   'minitaur8b': ROOT / 'models/minitaur8b',
                   'qwen35_4b': ROOT / 'models/qwen3.5-4b',
                   'intern_decision_4b': ROOT / 'models/intern-decision-4b',
                   'laya': ROOT / 'models/laya-english'}
    results = {}
    for name in models:
        directory = directories[name]
        config_path = directory / ('encoder/config.json' if name == 'laya' else 'config.json')
        cfg = json.loads(config_path.read_text())
        limit = cfg.get('text_config', cfg)['max_position_embeddings']
        tokenizer = AutoTokenizer.from_pretrained(
            directory / 'tokenizer' if name == 'laya' else directory, local_files_only=True)
        if name == 'laya':
            sys.path.insert(0, str(ROOT / 'external/laya'))
            from laya.common import build_sequence
            head_max_len = json.loads((directory / 'rl_agent_config.json').read_text())['head_max_len']
        elif name == 'intern_decision_4b':
            from examples.intern_decision_choice_agent import load_official_inference, choice_request
            official = load_official_inference()
            limit = 8192  # Existing official adapter's explicit encoder setting.
        else:
            formatter = object.__new__(FrozenDecoderChoice)
            formatter.tokenizer = tokenizer
            formatter.prompt_format = 'centaur_response_markers' if name == 'minitaur8b' else 'instruct_chat'
        lengths = []
        for trial in trials:
            keys = list(trial.output_keys)
            if name == 'laya':
                internal = {'t': 'choice', 'ins': trial.question['instructions'], 'crit': trial.question['criteria']}
                ids, markers = build_sequence(tokenizer, trial.state, internal, 1000000, head_max_len)
                if len(markers) != len(keys):
                    raise ValueError('All legal choices must survive Laya input construction.')
                n = len(ids)
            else:
                if name == 'intern_decision_4b':
                    compiled = official.compile_row(choice_request(trial.state, trial.question, keys))
                    prompt = tokenizer.apply_chat_template(compiled.messages, tokenize=False,
                        add_generation_prompt=False, enable_thinking=False, add_vision_id=True)
                elif name == 'qwen35_4b':
                    prompt, _ = format_choice_prompt(tokenizer, trial.state, trial.question, keys)
                else:
                    prompt, _ = formatter.format_prompt(trial.state, trial.question, keys)
                n = len(tokenizer.encode(prompt, add_special_tokens=False))
            lengths.append(n)
        ordered = sorted(lengths)
        results[name] = {'model_directory': str(directory), 'responses_checked': len(lengths),
            'minimum_input_tokens': min(lengths), 'median_input_tokens': statistics.median(lengths),
            'p95_input_tokens': ordered[math.ceil(.95 * len(ordered)) - 1],
            'maximum_input_tokens': max(lengths), 'sum_input_tokens': sum(lengths),
            'current_adapter_context_limit': limit,
            'responses_exceeding_current_context': sum(n > limit for n in lengths),
            'first_exceeding_id': next((t.id for t, n in zip(trials, lengths) if n > limit), None),
            'truncation': False, 'weights_loaded': False}
    return results
