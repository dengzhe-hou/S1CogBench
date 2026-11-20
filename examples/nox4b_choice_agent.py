"""Decision-2.0-Nox-4B through its own System One runtime, read like the accepted Jev rows.

The native choice is the action and the returned probabilities are the distribution. Over-budget
input is an error from the runtime, so nothing is truncated. The runtime needs at least two options,
so a question with one legal option gets probability 1 without a model call. The checkpoint's bundled code runs
with trust_remote_code at the pinned revision below.
"""
import os
from pathlib import Path
import time

from examples.templates.decision_api_choice import parse_choice, parse_event

PROBABILITY_READOUT = True
MODEL_ID = 'vllm-sr/Decision-2.0-Nox-4B'
REVISION = 'ae269ecf34a0b3a89026bc1afd9c995279212fad'
MODEL_DIR = Path(os.environ.get('S1COGBENCH_NOX_MODEL_DIR', 'models/decision-2.0-nox-4b'))


class Nox4BChoice:
    def __init__(self, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(MODEL_DIR), trust_remote_code=True, device=device, local_files_only=True)
        self.metadata = {'model_id': MODEL_ID, 'revision': REVISION, 'model_directory': str(MODEL_DIR),
                         'readout': 'Native typed choice is the action; returned probabilities; no floor.'}

    def ask(self, state, question):
        started = time.perf_counter()
        response = self.model.system_one(state=state, questions={'answer': question})
        self.last_seconds = time.perf_counter() - started
        self.last_input_tokens = response['usage']['input_tokens']
        self.last_raw_answer = response['answers']['answer']
        if 'error' in self.last_raw_answer:
            raise ValueError(f"Nox runtime error: {self.last_raw_answer['error']}")
        return self.last_raw_answer

    def choice_probabilities(self, state, question, output_keys):
        keys = list(output_keys)
        if len(keys) == 1:
            # The runtime rejects a question with one option. Its probability is 1 under any model.
            self.last_native_choice, self.last_raw_probabilities = keys[0], [1.0]
            self.last_raw_answer = {'type': 'choice', 'choice': keys[0], 'probabilities': {keys[0]: 1.0}, 'model_called': False}
            return [1.0]
        choice, raw, p = parse_choice(self.ask(state, question), keys)
        self.last_native_choice, self.last_raw_probabilities = choice, raw
        return p

    def posterior_probability(self, state, question):
        return parse_event(self.ask(state, question))

    def confidence_probability(self, state, chosen, original_specifications):
        return self.posterior_probability(state, {
            'type': 'noul', 'instructions': f'Machine {chosen} currently has the higher average reward.'})


def create_agent(device):
    return Nox4BChoice(device)
