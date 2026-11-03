"""Frozen LLM numeric reports for original CogBench V1 probability questions."""
import re

from src.hf_choice_models import FrozenDecoderChoice


def parse_probability_suffix(text):
    """Read the numeric completion after a prefilled 0.; never invent a report."""
    text = text.strip()
    text = re.sub(r'(?:>>?|»|`)?[.]?\s*$', '', text).strip()
    if re.fullmatch(r'[0-9]+', text):
        return float('0.' + text)
    if re.fullmatch(r'0\.[0-9]+', text):
        # The original parser also accepts a repeated 0.XX number.
        return float(text)
    raise ValueError(f'Invalid numerical probability completion: {text!r}; no random fallback.')


class FrozenNumericEvent:
    """Two generated tokens, greedy; q comes from text, not token probabilities."""

    def __init__(self, model_dir, device, prompt_format):
        self.base = FrozenDecoderChoice(model_dir, device, prompt_format)
        self.max_input_tokens = 0
        self.metadata = {
            'model_directory': str(self.base.model_dir), 'prompt_format': prompt_format,
            'actual_parameter_dtypes': self.base.metadata['actual_parameter_dtypes'],
            'training': 'Frozen checkpoint; no fitting, calibration or prompt selection.',
            'prior_training_overlap': 'Not ruled out; no unseen-task claim.',
            'event_semantics': 'Parse a numerical posterior report; no choice/token probability substituted.',
            'max_new_tokens': 2, 'do_sample': False,
            'temperature_status': 'Unused in greedy generation; no claim that checkpoint defaults were used.',
            'prompt': 'Original V1 visible story and query with a numerical 0. answer prefix. '
                      'Llama native chat template or Minitaur response markers are explicit adaptations.',
            'parser': 'Numeric suffix after 0., optionally repeated 0.XX, marker or backtick closure. '
                      'Invalid reports fail and remain incomplete; no original random fallback.',
            'parser_revision': 'Observed completions 5`, 5» and 5> represent 0.5 followed by markup; '
                               'accept those terminal format characters. Failed attempts remain saved.',
            'history': 'Independent questions; no private memory or reasoning tokens.',
            'original_decoder_status': 'Published two-token answer budget, with explicit model-native formatting '
                                       'and deterministic parsing adaptations; not an archived API rerun.',
            'model_generation_config': self.base.model.generation_config.to_dict(),
        }

    def posterior_probability(self, state, question):
        import torch

        target = re.search(r'urn ([FJ])', question['instructions']).group(1)
        color = re.search(r'A (red|blue)\s+ball was drawn\.', state).group(1)
        if self.base.prompt_format == 'instruct_chat':
            prompt = self.base.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': state}], tokenize=False, add_generation_prompt=True)
            prompt += f'I estimate the probability of the {color} ball to be drawn from the urn {target} to be 0.'
        else:
            prompt = state + '\nYou report <<0.'
        inputs = self.base.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        self.last_input_tokens = int(inputs.input_ids.shape[1])
        if self.last_input_tokens > self.base.model.config.max_position_embeddings:
            raise ValueError('Full probability question exceeds model context; do not truncate.')
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        with torch.inference_mode():
            generated = self.base.model.generate(
                **inputs.to(self.base.device), max_new_tokens=2, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.base.tokenizer.eos_token_id)
        tokens = generated.sequences[0, self.last_input_tokens:].tolist()
        selected_are_argmax = [token == int(scores[0].argmax())
                               for token, scores in zip(tokens, generated.scores)]
        if not all(selected_are_argmax) or len(tokens) > 2:
            raise ValueError('Actual generation does not match the declared two-token greedy budget.')
        suffix = self.base.tokenizer.decode(tokens, skip_special_tokens=True)
        self.last_raw_answer = {'generated_suffix': suffix, 'generated_token_ids': tokens,
                                'generated_token_count': len(tokens),
                                'chosen_tokens_are_argmax': selected_are_argmax}
        return parse_probability_suffix(suffix)
