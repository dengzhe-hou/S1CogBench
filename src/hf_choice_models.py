"""Frozen local BF16 decoders for the benchmark's explicit typed-choice track."""
import json
from pathlib import Path
import string

from src.choice_letter_tokens import ChoiceLetterTokens, IMPLEMENTATION

ROOT = Path(__file__).resolve().parents[1]


class FrozenDecoderChoice:
    """Temperature-one legal-token scoring; no reasoning, fitting or generation."""

    def __init__(self, model_dir, device, prompt_format):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('This BF16 8B comparison requires a CUDA device; queue when unavailable.')
        self.model_dir = Path(model_dir).resolve()
        self.prompt_format = prompt_format
        config = json.loads((self.model_dir / 'config.json').read_text())
        if config.get('quantization_config'):
            raise ValueError('Use the existing unquantized BF16 checkpoint for this comparison.')
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._choice_letter_tokens = ChoiceLetterTokens(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, local_files_only=True, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.max_input_tokens = 0
        self.metadata = {
            'model_directory': str(self.model_dir), 'prompt_format': prompt_format,
            'actual_parameter_dtypes': sorted({str(p.dtype) for p in self.model.parameters()}),
            'training': 'Frozen zero-shot; no task fitting, calibration or prompt tuning.',
            'probabilities': 'Softmax over the legal next-letter logits at temperature 1.',
            'native_choice': 'Argmax of these constrained probabilities; no free-text generation.',
            'interface': 'Explicit typed-choice adaptation; not the original published decoder.',
            'labels': 'Keep single-letter task keys. Otherwise use A,B,... in the given legal-action order.',
            'history': 'Full provided state without summarization or truncation.',
            'prior_training_overlap': 'Not ruled out; frozen deployment does not imply an unseen task.',
        }
        if self.prompt_format == 'centaur_response_markers':
            self.metadata.update(
                choice_tokenization_implementation=IMPLEMENTATION,
                choice_tokenization_timing='CPU suffix caching applies only to A-Z choices at the fixed You choose << marker, with full-prompt suffix checks included in adapter latency; numeric and other prompts retain per-candidate full-prompt tokenization; scalar model forward is unchanged.')

    def format_prompt(self, state, question, output_keys):
        labels = (list(output_keys) if all(len(k) == 1 and k in string.ascii_uppercase
                                         for k in output_keys)
                  else list(string.ascii_uppercase[:len(output_keys)]))
        if len(labels) != len(output_keys):
            raise ValueError('This benchmark adapter supports at most 26 legal actions.')
        options = '\n'.join(f'{label}: {key} - {question["criteria"][key]}'
                            for label, key in zip(labels, output_keys))
        content = state + '\n' + question['instructions'] + '\n' + options
        content += '\nChoose one of the listed option letters.'
        if self.prompt_format == 'instruct_chat':
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        elif self.prompt_format == 'centaur_response_markers':
            prompt = content + '\nYou choose <<'
        else:
            raise ValueError(f'Unknown prompt format: {self.prompt_format}')
        return prompt, labels

    def choice_probabilities(self, state, question, output_keys):
        import numpy as np
        import torch

        prompt, labels = self.format_prompt(state, question, output_keys)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        prefix = inputs['input_ids'][0].tolist()
        # IntegerLabelChoice shares the format name but uses a different marker/readout.
        if (self.prompt_format == 'centaur_response_markers'
                and prompt.endswith('\nYou choose <<')
                and all(len(label) == 1 and label in string.ascii_uppercase for label in labels)):
            token_ids = self._choice_letter_tokens.token_ids(prompt, prefix, labels)
        else:
            token_ids = []
            for label in labels:
                joined = self.tokenizer.encode(prompt + label, add_special_tokens=False)
                if joined[:-1] != prefix:
                    raise ValueError('Each legal output must be one next token after the actual prompt.')
                token_ids.append(joined[-1])
        if len(set(token_ids)) != len(output_keys):
            raise ValueError('Legal outputs must have distinct token ids.')
        self.last_input_tokens = len(prefix)
        if self.last_input_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Full history exceeds the model context; no silent truncation.')
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        self.last_label_mapping = dict(zip(labels, output_keys))
        with torch.inference_mode():
            logits = self.model(**inputs.to(self.device), use_cache=False,
                                logits_to_keep=1).logits[0, -1].float()
        selected = logits[token_ids].double()
        if not torch.isfinite(selected).all():
            raise ValueError('Non-finite legal choice logits.')
        probability = torch.softmax(selected, -1).cpu().numpy()
        probability /= probability.sum()
        self.last_allowed_token_mass = float(torch.exp(
            torch.logsumexp(logits[token_ids], 0) - torch.logsumexp(logits, 0)))
        self.last_native_choice = output_keys[int(np.argmax(probability))]
        return probability

    def choice_probabilities_batch(self, requests, batch_size=8):
        """Independent full-history requests; returns per-request batch metadata."""
        from src.batched_choices import choice_probabilities_batch

        return choice_probabilities_batch(self, requests, batch_size=batch_size)
