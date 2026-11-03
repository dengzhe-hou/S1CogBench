"""Frozen Qwen3-14B Base / Qwentaur-r64 reports on CogBench urn questions.

This is a cross-task generation adapter, not Centauri's published likelihood
evaluation. Both checkpoints receive the same original CogBench V1 answer
prefix and the published two-token greedy answer budget.
"""
import re

from src.numeric_event_models import parse_probability_suffix


class Qwen14BNumericEvent:
    def __init__(self, device, model_name):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from scripts.run_psych_official_local import (
            ADAPTER, BASE, check_assets, dtype_record, tokenizer_for,
        )

        if model_name not in {'qwentaur14b-r64', 'qwen14b-base'}:
            raise ValueError(f'Unknown 14B checkpoint: {model_name}')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('The original BF16 14B checkpoints require a scheduled CUDA device.')
        check_assets()
        torch.set_num_threads(4)
        self.tokenizer = tokenizer_for(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            BASE, local_files_only=True, dtype=torch.bfloat16).to(self.device)
        self.model = (PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
                      if model_name == 'qwentaur14b-r64' else base).eval()
        self.max_input_tokens = 0
        self.metadata = {
            'checkpoint': model_name,
            'base_directory': str(BASE),
            'adapter_directory': str(ADAPTER) if model_name == 'qwentaur14b-r64' else None,
            **dtype_record(self.model, model_name),
            'actual_parameter_dtypes': sorted({str(p.dtype) for p in self.model.parameters()}),
            'training': 'Frozen; no CogBench fitting, calibration or prompt selection.',
            'prior_training_overlap': 'Not ruled out; no unseen-task claim.',
            'prompt': 'Original CogBench V1 visible question and exact A: numerical 0. answer prefix.',
            'prompt_format': 'original_cogbench_Q_A_prefix_two_token_numeric_adaptation',
            'decoding': 'Original two generated tokens, greedy; parse numerical report, not token probability.',
            'decoder_status': 'Cross-task adapter to CogBench, not a published Centauri direct-report evaluation.',
            'invalid_report_policy': 'Fail and preserve the incomplete run; no random report substituted.',
            'model_generation_config': self.model.generation_config.to_dict(),
        }

    @staticmethod
    def format_prompt(state, question):
        target = re.search(r'urn ([FJ])', question['instructions']).group(1)
        color = re.search(r'A (red|blue)\s+ball was drawn\.', state).group(1)
        return (state + '\n\nA: ' +
                f'I estimate the probability of the {color} ball to be drawn '
                f'from the urn {target} to be 0.')

    def posterior_probability(self, state, question):
        import torch

        prompt = self.format_prompt(state, question)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        self.last_input_tokens = int(inputs.input_ids.shape[1])
        if self.last_input_tokens + 2 > self.model.config.max_position_embeddings:
            raise ValueError('Full original question plus answer budget exceeds model context.')
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs.to(self.device), max_new_tokens=2, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id)
        tokens = generated.sequences[0, self.last_input_tokens:].tolist()
        selected_are_argmax = [token == int(scores[0].argmax())
                               for token, scores in zip(tokens, generated.scores)]
        if not all(selected_are_argmax) or len(tokens) > 2:
            raise ValueError('Actual generation does not match the two-token greedy budget.')
        suffix = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.last_raw_answer = {
            'generated_suffix': suffix,
            'generated_token_ids': tokens,
            'generated_token_count': len(tokens),
            'chosen_tokens_are_argmax': selected_are_argmax,
        }
        return parse_probability_suffix(suffix)
