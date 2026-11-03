"""Template: a text-generation API (Anthropic Messages) read through the retained FINAL line.

This is the readout of the accepted Claude rows (runs/api_small_tasks_20261030
and the earlier Claude runs it reuses):

- Request: the Messages API with max_tokens 4096, adaptive summarized thinking
  and medium effort, one user message: the visible state, the question's
  instructions, one line "key: criterion" per legal option, and INSTRUCTION.
- Parser: the strict FINAL-line parser of
  runs/probes_20261028/p2_completion_20261028/generation_agent.py,
  read from text blocks only (thinking is kept apart). Exactly one line
  "FINAL: X" with a legal X, as the last line; a reply that stops for any
  reason other than end_turn is invalid. Generalized to legal keys: single
  capital letters keep the retained class [A-Z], single digits the class [0-9]
  of the accepted digit-key runs, multi-digit integer keys use [0-9]+, and any
  other key set takes the rest of the FINAL line. No fallback, no retry.
- No action probabilities are inferred from text. choice_probabilities returns
  a one-hot vector on the parsed choice, so the runners' action and accuracy
  rules use it. PROBABILITY_READOUT = False tells the CLI to leave NLL targets
  undefined (n.a. in Table 3).
- An invalid reply sets last_native_choice to "INVALID": prediction scorers
  count it as an incorrect, invalid response; autonomous runners reject it and
  the mode stops, as the accepted runs ended that trajectory.
- posterior_probability and confidence_probability ask for one line
  "FINAL: <number>" with a probability from 0 to 1 (retained Claude Bayesian
  prompt and parser); an invalid reply returns None.

The key is read from ANTHROPIC_API_KEY in the environment only and never
written anywhere. Every request is billed, including thinking tokens.
"""
import json
import os
import re
import time
import urllib.request

PROBABILITY_READOUT = False
URL = 'https://api.anthropic.com/v1/messages'
MODEL = os.environ.get('S1_GENERATIVE_MODEL', 'claude-haiku-5-5')
NATIVE = {'max_tokens': 4096, 'thinking': {'type': 'adaptive', 'display': 'summarized'},
          'output_config': {'effort': 'medium'}}
INSTRUCTION = (
    'Reason about the decision using only the information shown above. '
    'You may write reasoning before the answer. '
    'End your response with exactly one final line in this format: FINAL: X\n'
    'Replace X with one of the listed legal option {noun}. '
    'Do not put any text after that line.'
)
NUMBER_INSTRUCTION = ('\nYou may reason before answering. End your response with exactly one line '
                      'FINAL: <number>, where <number> is a probability from 0 to 1. '
                      'Do not include another FINAL marker in the final text.')
INVALID = 'INVALID'


def key_class(keys):
    if all(len(k) == 1 and k.isupper() for k in keys):
        return r'[A-Z]', 'letters'
    if all(len(k) == 1 and k.isdigit() for k in keys):
        return r'[0-9]', 'numbers'
    if all(k.isdigit() for k in keys):
        return r'[0-9]+', 'numbers'
    return r'.+?', 'keys'


def content_for(state, question, keys):
    if len(keys) != len(set(keys)) or set(keys) != set(question['criteria']):
        raise ValueError('Question criteria must exactly match distinct legal output keys.')
    options = '\n'.join(f"{k}: {question['criteria'][k]}" for k in keys)
    return (state + '\n' + question['instructions'] + '\n' + options + '\n'
            + INSTRUCTION.format(noun=key_class(keys)[1]))


def parse_answer(text, keys, truncated):
    """Retained strict final-line parser with the key class generalized to the legal keys."""
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return {'choice': None, 'invalid': True,
                'invalid_reason': 'max_new_tokens_reached' if truncated else 'unclosed_native_thinking'}
    pattern = key_class(keys)[0]
    matches = re.findall(r'^\s*FINAL:\s*(' + pattern + r')\s*$', text.strip(), flags=re.MULTILINE)
    choice = matches[0] if len(matches) == 1 and matches[0] in keys else None
    if choice is not None and not re.search(r'FINAL:\s*' + re.escape(choice) + r'\s*$', text.strip()):
        choice = None
    return {'choice': None if truncated else choice, 'invalid': bool(truncated or choice is None),
            'invalid_reason': 'max_new_tokens_reached' if truncated else
                ('missing_ambiguous_or_illegal_final_line' if choice is None else None)}


def parse_number(text, stop_reason):
    """Retained numeric FINAL parser of the Claude Bayesian runs."""
    hits = re.findall(r'FINAL\s*:\s*([^\n]*)', text)
    m = re.search(r'(?:^|\n)\s*FINAL:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\Z', text)
    return float(m.group(1)) if stop_reason == 'end_turn' and len(hits) == 1 and m else None


class GenerativeAPIChoice:
    def __init__(self, model=MODEL, post=None):
        key = os.environ.get('ANTHROPIC_API_KEY')
        if post is None and not key:
            raise SystemExit('Set ANTHROPIC_API_KEY in the environment.')
        self._key, self.model = key, model
        self._post = post or self._http
        self.metadata = {'endpoint': URL, 'model': model, 'native_config': NATIVE,
                         'instruction': INSTRUCTION, 'parser': 'Exactly one legal FINAL line at the end of the '
                         'text blocks; non-end_turn stop invalid; no fallback or retry.',
                         'action_probabilities': 'none (one-hot on the parsed choice)', 'automatic_retries': 0}

    def _http(self, body):
        request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            'x-api-key': self._key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def generate(self, content):
        started = time.perf_counter()
        response = self._post({'model': self.model, **NATIVE, 'messages': [{'role': 'user', 'content': content}]})
        self.last_seconds = time.perf_counter() - started
        text = '\n'.join(b['text'] for b in response['content'] if b['type'] == 'text')
        self.last_raw_answer = {'text': text, 'stop_reason': response['stop_reason'],
                                'id': response.get('id'), 'usage': response.get('usage')}
        self.last_input_tokens = (response.get('usage') or {}).get('input_tokens')
        return text, response['stop_reason']

    def choice_probabilities(self, state, question, output_keys):
        keys = list(output_keys)
        if not isinstance(state, str):
            raise ValueError('This template expects a text state.')
        text, stop = self.generate(content_for(state, question, keys))
        parsed = parse_answer(text, keys, stop in ('max_tokens', 'model_context_window_exceeded'))
        if stop != 'end_turn':
            parsed.update(choice=None, invalid=True, invalid_reason='provider_stop_' + str(stop))
        self.last_parse = parsed
        if parsed['invalid']:
            self.last_native_choice = INVALID
            return [1.0 / len(keys)] * len(keys)
        self.last_native_choice = parsed['choice']
        return [float(k == parsed['choice']) for k in keys]

    def posterior_probability(self, state, question):
        text, stop = self.generate(state + '\n\nReport the probability that this statement is true: '
                                   + question['instructions'] + NUMBER_INSTRUCTION)
        return parse_number(text, stop)

    def confidence_probability(self, state, chosen, original_specifications):
        text, stop = self.generate(state + NUMBER_INSTRUCTION)
        return parse_number(text, stop)


def create_agent(device):
    return GenerativeAPIChoice()
