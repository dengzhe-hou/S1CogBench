"""Use the full original integer labels with Llama-family one-token readouts.

No bins or digit subquestions. The base scorer verifies every label after the
actual complete prompt before taking a softmax over legal next-token logits.
"""
from src.hf_choice_models import FrozenDecoderChoice


class IntegerLabelChoice(FrozenDecoderChoice):
    def format_prompt(self, state, question, output_keys):
        if not all(str(k).isdigit() for k in output_keys):
            return super().format_prompt(state, question, output_keys)
        labels = list(output_keys)
        content = state + '\n' + question['instructions']
        content += '\nLegal integer reports: ' + ', '.join(labels) + '.'
        content += '\nReturn only one legal integer.'
        if self.prompt_format == 'instruct_chat':
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        elif self.prompt_format == 'centaur_response_markers':
            prompt = content + '\nYou report <<'
        else:
            raise ValueError(self.prompt_format)
        return prompt, labels

    def __init__(self, model_dir, device, prompt_format):
        super().__init__(model_dir, device, prompt_format)
        self.metadata.update(integer_interface='All original numeric labels directly scored as distinct next tokens; no letter mapping, bins, or digit subdecisions.',
                             labels='Keep original single-letter answer keys and full original integer labels; numeric labels do not use the 26-letter map.',
                             probabilities='Softmax over the complete legal next-token logits at temperature 1.',
                             integer_boundary_check='Every candidate is tokenized after every complete actual prompt; fail if not exactly one token.',
                             readout_temperature=1,
                             interface='Added typed numeric-label prompt; not archived original full-vocabulary NLL.')
