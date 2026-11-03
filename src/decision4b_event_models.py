"""Explicit event reports for the frozen Intern-Decision and Qwen 4B models."""
import math
import re

from src.numeric_event_models import parse_probability_suffix


def posterior_prefix(state, question):
    target = re.search(r'urn ([FJ])', question['instructions']).group(1)
    color = re.search(r'A (red|blue)\s+ball was drawn\.', state).group(1)
    return f'I estimate the probability of the {color} ball to be drawn from the urn {target} to be 0.'


def confidence_question(chosen):
    return {'type': 'noul', 'instructions':
            f'Your selected machine {chosen} currently has the higher average reward. '
            'Report the probability that this statement is true using only your observations.'}


def numeric_prompt(tokenizer, content, prefix):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) + prefix


class InternDecisionEvent:
    def __init__(self, device):
        from examples.intern_decision_choice_agent import create_agent
        self.choice = create_agent(device)
        self.max_input_tokens = 0
        self.metadata = {**self.choice.metadata,
            'event_interface': 'Official noul report for the explicitly stated event; default checkpoint temperature.',
            'probability_semantics': 'noul is the probability of the stated event, not action confidence.',
            'confidence': 'Independent noul query about the selected machine currently having higher mean reward.',
            'shared_weights': 'Action and event queries share the same frozen checkpoint.',
            'original_numeric_decoder': 'Explicit native event adaptation, not original two-token generation.'}

    def posterior_probability(self, state, question):
        if question['type'] != 'noul':
            raise ValueError('Expected an explicit event/noul question.')
        request = {'state': state, 'questions': {'event': question}}
        response = self.choice.engine.predict(request)
        answer = response['answers']['event']
        probability = float(answer['noul'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Official event readout must be a probability.')
        self.last_input_tokens = int(response['usage']['input_tokens'])
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_raw_answer = answer
        compiled = self.choice.official.compile_row(request)
        self.last_prompt = self.choice.engine.tokenizer.apply_chat_template(
            compiled.messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True)
        return probability

    def choice_probabilities(self, state, question, keys):
        probability = self.choice.choice_probabilities(state, question, keys)
        self.last_native_choice = self.choice.last_native_choice
        self.last_input_tokens = self.choice.last_input_tokens
        self.last_prompt = self.choice.last_prompt
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        return probability

    def confidence_probability(self, prompt, chosen, specifications):
        return self.posterior_probability(prompt + specifications, confidence_question(chosen))


class Qwen35NumericEvent:
    def __init__(self, device):
        from examples.qwen35_4b_choice_agent import create_agent
        self.choice = create_agent(device)
        self.max_input_tokens = 0
        self.metadata = {**self.choice.metadata,
            'event_interface': 'Two-token greedy numerical report after the original 0. prefix.',
            'probability_semantics': 'Parse the reported number; never substitute a selected-token/action probability.',
            'confidence': 'Independent numerical confidence report after selected-machine 0. prefix.',
            'max_new_tokens': 2, 'do_sample': False,
            'shared_weights': 'Action and numerical report share the same frozen official checkpoint.',
            'numeric_parser': 'Existing original-protocol numeric suffix parser; raw tokens saved, invalid reports fail.',
            'original_numeric_decoder': 'Original two-token greedy budget with explicit native non-thinking chat formatting.'}

    def numerical_probability(self, content, prefix):
        import torch
        base = self.choice
        prompt = numeric_prompt(base.tokenizer, content, prefix)
        inputs = base.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        self.last_input_tokens = int(inputs.input_ids.shape[1])
        if self.last_input_tokens > base.max_context_tokens:
            raise ValueError('Complete event-report input exceeds the checkpoint context; no truncation.')
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        with torch.inference_mode():
            generated = base.model.generate(
                **inputs.to(base.device), max_new_tokens=2, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=base.tokenizer.eos_token_id)
        tokens = generated.sequences[0, self.last_input_tokens:].tolist()
        argmax = [token == int(score[0].argmax())
                  for token, score in zip(tokens, generated.scores)]
        suffix = base.tokenizer.decode(tokens, skip_special_tokens=True)
        self.last_raw_answer = {'generated_suffix': suffix, 'generated_token_ids': tokens,
                               'generated_token_count': len(tokens), 'chosen_tokens_are_argmax': argmax}
        if len(tokens) > 2 or len(argmax) != len(tokens) or not all(argmax):
            raise ValueError('Actual generation differs from the two-token greedy protocol.')
        return parse_probability_suffix(suffix)

    def posterior_probability(self, state, question):
        return self.numerical_probability(state, posterior_prefix(state, question))

    def choice_probabilities(self, state, question, keys):
        probability = self.choice.choice_probabilities(state, question, keys)
        self.last_native_choice = self.choice.last_native_choice
        self.last_input_tokens = self.choice.last_input_tokens
        self.last_prompt = self.choice.last_prompt
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        return probability

    def confidence_probability(self, prompt, chosen, specifications):
        return self.numerical_probability(prompt + specifications,
                                          f'Machine {chosen} with confidence 0.')
