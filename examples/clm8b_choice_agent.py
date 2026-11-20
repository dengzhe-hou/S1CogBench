"""CLM-v0.1-8B: official heads, Engine and Embedder, with the official Qwen3-8B vLLM pooling server.

The official Embedder asks the server for the truncated prompt, 2,048 tokens per text, so longer
text is dropped as in the official setup. Each call records whether its state was truncated.
"""
import atexit
import json
import os
from pathlib import Path
import time

from clm import Engine
from clm.embedder import Embedder
from clm.schema import state_text

from examples.templates.decision_api_choice import parse_choice, parse_event

PROBABILITY_READOUT = True
MODEL_ID = 'Contrastive-LM/CLM-v0.1-8B'
REVISION = 'e939398d4556fcd9400c76fa8c5a513202f42b0a'
ENCODER_ID = 'Qwen/Qwen3-8B'
ENCODER_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
HEAD_FILE = Path(os.environ.get('S1COGBENCH_CLM_HEAD', 'models/clm-v0.1-8b/CLM_v0.1-8B.pt'))
ENCODER_DIR = Path(os.environ.get('S1COGBENCH_QWEN8B_DIR', 'models/qwen3-8b'))
EMBEDDINGS_URL = os.environ.get('S1COGBENCH_CLM_EMB_URL', 'http://127.0.0.1:8090/v1/embeddings')
MAX_TOKENS = 2048
# Unset is the official setup: vLLM's pooling runner then keeps the first 2,048 tokens, which drops the
# question at the end of a long state. 'left' keeps the last 2,048 tokens, for a separately labeled control.
TRUNCATION_SIDE = os.environ.get('S1COGBENCH_CLM_TRUNCATION_SIDE')


class SidedEmbedder(Embedder):
    def __init__(self, side, **kwargs):
        super().__init__(**kwargs)
        post = self.session.post
        self.session.post = lambda url, json, **kw: post(url, json={**json, 'truncation_side': side}, **kw)


class CLM8BChoice:
    def __init__(self, device):
        self.calls, self.truncated_calls, self.max_state_tokens = 0, 0, 0
        if os.environ.get('S1COGBENCH_CLM_TRUNCATION_LOG'):
            atexit.register(self.write_truncation_log)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(ENCODER_DIR), local_files_only=True)
        kwargs = dict(url=EMBEDDINGS_URL, model='qwen3-8b', max_tokens=MAX_TOKENS)
        embedder = SidedEmbedder(TRUNCATION_SIDE, **kwargs) if TRUNCATION_SIDE else Embedder(**kwargs)
        self.engine = Engine(embedder=embedder, checkpoint=str(HEAD_FILE), device=device)
        truncation = (f'2,048 tokens per text, truncation_side={TRUNCATION_SIDE} (control, not the official setup)'
                      if TRUNCATION_SIDE else 'official server truncation to 2,048 tokens per text')
        self.metadata = {'model_id': MODEL_ID, 'revision': REVISION, 'encoder': ENCODER_ID, 'encoder_revision': ENCODER_REVISION,
                         'readout': f'Native typed choice is the action; softmax over scaled cosine; {truncation}.'}

    def ask(self, state, question):
        started = time.perf_counter()
        answer = self.engine.answer(state, {'answer': question})['answers']['answer']
        self.last_seconds = time.perf_counter() - started
        tokens = len(self.tokenizer(state_text(state, question.get('instructions')))['input_ids'])
        self.last_input_tokens = tokens
        self.calls += 1
        self.truncated_calls += tokens > MAX_TOKENS
        self.max_state_tokens = max(self.max_state_tokens, tokens)
        self.last_raw_answer = {**answer, 'state_tokens_before_truncation': tokens, 'truncated': tokens > MAX_TOKENS}
        return answer

    def write_truncation_log(self):
        Path(os.environ['S1COGBENCH_CLM_TRUNCATION_LOG']).write_text(json.dumps(
            {'calls': self.calls, 'truncated_calls': self.truncated_calls, 'max_state_tokens': self.max_state_tokens, 'limit': MAX_TOKENS}))

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
    return CLM8BChoice(device)
