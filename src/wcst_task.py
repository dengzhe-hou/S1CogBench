"""Original WCST test split, prompts and error definitions from Namazova et al."""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/paper_matched_centaur_evaluation/wcst'
LETTERS = 'ABCD'
QUESTION = {'type': 'choice', 'instructions': 'Which key do you press?',
            'criteria': dict(zip(LETTERS, ['one red triangle', 'two green stars',
                                          'three yellow crosses', 'four blue balls']))}


def test_data():
    data = np.load(SOURCE / 'Data/json_wcst_dataset.npy', allow_pickle=True)
    ids = np.unique([r['subject_id'] for r in data])
    test_ids = ids[:int(len(ids) * .2)]
    by_subject = defaultdict(list)
    for row in data:
        if row['subject_id'] in test_ids:
            by_subject[int(row['subject_id'])].append(row)
    assert len(by_subject) == 75 and sum(map(len, by_subject.values())) == 12754
    return dict(sorted(by_subject.items()))


def prompt_helpers(family, mode):
    """Extract actual official constants/functions without running their GPU entrypoint."""
    filename = f'{mode}_{family}_70B.py'
    path = SOURCE / 'generative_predictive_llms_eval' / filename
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    constant = 'START_MSG' if family == 'llama' else 'start_prompt'
    start = next(ast.literal_eval(n.value) for n in main.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
    names = {'key_equal', 'step_to_user_msg', 'feedback_msg'} if family == 'llama' else {
        'step_to_prompt', 'finish_step_prompt'}
    helpers = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'LETTER2NUM': dict(zip(LETTERS, range(1, 5))),
                 'NUM2LETTER': dict(enumerate(LETTERS, 1))}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), namespace)
    return start, namespace


class Dialogue:
    def __init__(self, family, mode):
        self.family, self.mode = family, mode
        start, self.fn = prompt_helpers(family, mode)
        self.history = [start] if family == 'llama' else start.strip() + '\n\n'

    def prompt(self, row, tokenizer=None):
        if self.family == 'llama':
            messages = self.history + [self.fn['step_to_user_msg'](row)]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.history + self.fn['step_to_prompt'](row)

    def update(self, row, key, letter=None):
        if self.family == 'llama':
            letter = letter if letter is not None else LETTERS[key - 1]
            self.history.extend([self.fn['step_to_user_msg'](row), self.fn['feedback_msg'](row, letter)])
        else:
            if key is None:
                raise ValueError('Official Centaur generative formatter cannot accept an invalid key; no fallback action.')
            answer = LETTERS[key - 1] if self.mode == 'generative' else key
            self.history += self.fn['step_to_prompt'](row) + self.fn['finish_step_prompt'](row, answer) + '\n'


def rule(row, key):
    if key is None:
        return None
    card = row['key_cards'][key]
    for i, name in enumerate(['color', 'form', 'number']):
        if card[i] == row['stimulus'][i]:
            return name
    return None


def task_metrics(rows, keys):
    correct, persev, setloss = [], [], []
    prev_ok, prev_rule = None, None
    for row, key in zip(rows, keys):
        ok, current = int(key == row['ground_key']), rule(row, key)
        correct.append(ok)
        persev.append(int(prev_ok == 0 and ok == 0 and current == prev_rule))
        setloss.append(int(prev_ok == 1 and ok == 0 and current != prev_rule))
        prev_ok, prev_rule = ok, current
    return {'accuracy': float(np.mean(correct)), 'perseveration_error': float(np.mean(persev)),
            'set_loss_error': float(np.mean(setloss))}
