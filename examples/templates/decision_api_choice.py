"""Template: a decision API in the Jev style as an S1CogBench adapter.

The API receives the visible state and the typed question and returns a
native answer. This is the readout of the accepted Jev 1.14 rows
(runs/api_small_tasks_20261030, runs/api_big_tasks_20261030):

- Request: POST {"model": ..., "state": state, "questions": {"answer": question}}
  to S1_DECISION_API_URL with "Authorization: Bearer $S1_DECISION_API_KEY".
- Choice questions: the native choice is the action. The returned two-decimal
  probabilities over the legal keys are divided by their returned sum, which
  must lie within 0.005 per option of 1. No floor and no smoothing: a returned
  0.00 stays 0, so a human choice given 0.00 has infinite strict NLL.
- Event questions (type "noul": Bayesian urn, nonliteral numbers): the
  returned probability is the report. Restless confidence is the event query
  "Machine <chosen> currently has the higher average reward.".
- No retries. A transport failure, an illegal choice or probabilities outside
  the rounding bound raise an error and the mode stops; nothing is replaced.

The key is read from the environment only and never written anywhere.
Every request is billed by the provider. Try a few calls first:

    export S1_DECISION_API_KEY=...        # do not put it in a file under the repo
    python -m s1cogbench eval --smoke --adapter my_decision_agent.py --output runs/api-smoke
"""
import json
import math
import os
import time
import urllib.request

PROBABILITY_READOUT = True
URL = os.environ.get('S1_DECISION_API_URL', 'https://openrouter.ai/api/alpha/decisions')
MODEL = os.environ.get('S1_DECISION_MODEL', 'typesafe/jev-1.14')
TIMEOUT_SECONDS = 180


class InvalidDecision(ValueError):
    pass


def parse_choice(answer, keys):
    """Native choice and legal probabilities: raw two-decimal values over their returned sum."""
    if not isinstance(answer, dict) or answer.get('type') != 'choice':
        raise InvalidDecision('no native choice answer')
    if answer.get('choice') not in keys:
        raise InvalidDecision(f'native choice {answer.get("choice")!r} is not legal')
    probabilities = answer.get('probabilities')
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise InvalidDecision('probability keys differ from the legal options')
    raw = [float(probabilities[k]) for k in keys]
    total = math.fsum(raw)
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in raw) or total <= 0:
        raise InvalidDecision('probability outside [0, 1]')
    if abs(total - 1) > len(keys) * .005 + 1e-8:
        raise InvalidDecision(f'probabilities sum to {total}, outside the two-decimal rounding bound')
    return answer['choice'], raw, [p / total for p in raw]


def parse_event(answer):
    if not isinstance(answer, dict) or answer.get('type') != 'noul':
        raise InvalidDecision('no native event-probability answer')
    q = float(answer['noul'])
    if not (math.isfinite(q) and 0 <= q <= 1):
        raise InvalidDecision(f'event probability {q} outside [0, 1]')
    return q


class DecisionAPIChoice:
    def __init__(self, url=URL, model=MODEL, post=None):
        key = os.environ.get('S1_DECISION_API_KEY')
        if post is None and not key:
            raise SystemExit('Set S1_DECISION_API_KEY in the environment.')
        self._key = key
        self.url, self.model = url, model
        self._post = post or self._http
        self.metadata = {'endpoint': url, 'requested_model': model,
                         'readout': 'Native typed choice is the action; raw two-decimal probabilities '
                                    'divided by their returned sum within .005 per option; no floor.',
                         'automatic_retries': 0}

    def _http(self, body):
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)

    def ask(self, state, question):
        started = time.perf_counter()
        response = self._post({'model': self.model, 'state': state, 'questions': {'answer': question}})
        self.last_seconds = time.perf_counter() - started
        usage = response.get('usage', {})
        self.last_input_tokens = usage.get('input_tokens')
        self.last_response_meta = {'id': response.get('id'), 'model': response.get('model'),
                                   'provider': response.get('provider'), 'usage': usage}
        answers = response.get('answers')
        self.last_raw_answer = answers.get('answer') if isinstance(answers, dict) else None
        return self.last_raw_answer

    def choice_probabilities(self, state, question, output_keys):
        keys = list(output_keys)
        choice, raw, p = parse_choice(self.ask(state, question), keys)
        self.last_native_choice, self.last_raw_probabilities = choice, raw
        return p

    def posterior_probability(self, state, question):
        return parse_event(self.ask(state, question))

    def confidence_probability(self, state, chosen, original_specifications):
        return self.posterior_probability(state, {
            'type': 'noul', 'instructions': f'Machine {chosen} currently has the higher average reward.'})


def create_agent(device):
    return DecisionAPIChoice()
