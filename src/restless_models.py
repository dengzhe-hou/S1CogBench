"""Use the existing frozen weights for action and a separate confidence report."""
import math

import torch

from src.numeric_event_models import parse_probability_suffix


class LayaRestless:
    def __init__(self, device):
        from examples.laya_choice_agent import create_agent
        self.choice = create_agent(device)
        self.choice.agent.cfg['max_len'] = 4096
        self.max_input_tokens = 0
        self.metadata = {'model': 'Laya-English', 'precision': 'FP32',
                         'max_len': 4096, 'head_max_len': self.choice.agent.cfg['head_max_len'],
                         'input_budget_status': 'Official runtime budget override, complete 80-trial history.',
                         'confidence': 'Separate native noul for chosen-machine currently-best event; not action p.',
                         'training': 'Frozen public checkpoint, native temperatures, no task fitting.'}

    def choice_probabilities(self, state, question, keys):
        p = self.choice.choice_probabilities(state, question, keys)
        self.last_native_choice = self.choice.last_native_choice
        self.last_input_tokens = self.choice.last_input_tokens
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        return p

    def confidence_probability(self, prompt, chosen, specifications):
        sdk = self.choice.agent
        question = {'type': 'noul', 'instructions':
                    f'Your selected machine {chosen} currently has the higher average reward. '
                    'Report the probability that this statement is true using only your observations.'}
        internal = sdk._to_internal(question)
        state = prompt + specifications
        ids, markers = self.choice.build_sequence(sdk.tok, state, internal,
                                                  sdk.cfg['max_len'], sdk.cfg['head_max_len'])
        full_ids, full_markers = self.choice.build_sequence(sdk.tok, state, internal, 8192, 8192)
        if ids != full_ids or markers != full_markers:
            raise ValueError('Restless event input was truncated.')
        self.last_input_tokens = len(ids)
        self.max_input_tokens = max(self.max_input_tokens, len(ids))
        answer = sdk.predict(state, {'event': question})['answers']['event']
        q = float(answer['noul'])
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Invalid SDK confidence-event probability.')
        self.last_raw_answer = answer
        return q


class DecoderRestless:
    def __init__(self, choice):
        self.choice = choice
        self.max_input_tokens = 0
        self.metadata = {**choice.metadata, 'confidence': 'Independent two-token greedy numeric report '
                         'after selected-machine confidence 0. prefix; not chosen-action probability.',
                         'numeric_parser': 'Numeric suffix; observed terminal >>, >, » or backtick markup removed. '
                                           'Raw suffix retained; invalid numeric content still fails.',
                         'shared_weights': 'Choice and confidence calls reuse the same frozen model.'}

    def choice_probabilities(self, state, question, keys):
        p = self.choice.choice_probabilities(state, question, keys)
        self.last_native_choice = self.choice.last_native_choice
        self.last_input_tokens = self.choice.last_input_tokens
        self.last_prompt = self.choice.last_prompt
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        return p

    def confidence_probability(self, prompt, chosen, specifications):
        base = self.choice
        content = prompt + specifications
        if base.prompt_format == 'instruct_chat':
            rendered = base.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            rendered += f'Machine {chosen} with confidence 0.'
        else:
            rendered = content + f'\nYou report <<Machine {chosen} with confidence 0.'
        inputs = base.tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
        self.last_input_tokens = inputs.input_ids.shape[1]
        if self.last_input_tokens > base.model.config.max_position_embeddings:
            raise ValueError('Complete Restless history exceeds decoder context.')
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = rendered
        with torch.inference_mode():
            generated = base.model.generate(**inputs.to(base.device), max_new_tokens=2, do_sample=False,
                                             return_dict_in_generate=True, output_scores=True,
                                             pad_token_id=base.tokenizer.eos_token_id)
        tokens = generated.sequences[0, self.last_input_tokens:].tolist()
        argmax = [token == int(score[0].argmax()) for token, score in zip(tokens, generated.scores)]
        suffix = base.tokenizer.decode(tokens, skip_special_tokens=True)
        self.last_raw_answer = {'suffix': suffix, 'token_ids': tokens,
                               'generated_token_count': len(tokens), 'selected_are_argmax': argmax}
        if not all(argmax) or len(tokens) > 2:
            raise ValueError('Confidence generation exceeded its original two-token greedy budget.')
        return parse_probability_suffix(suffix)
