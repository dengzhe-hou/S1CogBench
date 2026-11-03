"""Frozen scorers for a common state + ordered legal-choice interface (CPU by default)."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class LayaChoice:
    def __init__(self, device='cpu'):
        sys.path.insert(0, str(ROOT / 'external/laya'))
        import laya
        from laya.common import build_sequence
        self.agent = laya.load(str(ROOT / 'models/laya-english'), device=device)
        self.requested_device = torch.device(device)
        if self.agent.device != self.requested_device:
            raise RuntimeError(f'Laya used {self.agent.device}, requested {device}.')
        self.build_sequence = build_sequence
        self.max_input_tokens = 0

    def choice_probabilities(self, state, question, output_keys):
        internal = {'t': 'choice', 'ins': question['instructions'], 'crit': question['criteria']}
        cfg = self.agent.cfg
        ids, markers = self.build_sequence(self.agent.tok, state, internal,
                                           cfg['max_len'], cfg['head_max_len'])
        full_ids, full_markers = self.build_sequence(self.agent.tok, state, internal, 4096, 4096)
        assert ids == full_ids and markers == full_markers, 'State or choices truncated.'
        assert len(markers) == len(output_keys)
        self.last_input_tokens = len(ids)
        self.max_input_tokens = max(self.max_input_tokens, len(ids))
        answer = self.agent.predict(state, {'action': question})['answers']['action']
        if self.agent.device != self.requested_device:
            raise RuntimeError(f'Laya changed device to {self.agent.device}; timing is not comparable.')
        raw = np.array([answer['probabilities'][key] for key in output_keys], dtype=float)
        assert np.isfinite(raw).all() and (raw >= 0).all() and raw.sum() > 0
        # SDK rounds each probability to 4 decimals. Preserve the returned values
        # and explicitly normalize them for sampling over >2 legal actions.
        assert abs(raw.sum() - 1) <= len(raw) * 0.00005 + 1e-8
        self.last_raw_probabilities = raw.tolist()
        self.last_native_choice = answer['choice']
        return raw / raw.sum()


class SmolLMChoice:
    """One forward pass, temperature 1, constrained next-letter probabilities."""
    def __init__(self, path=None, device='cpu'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = path or ROOT / 'models/SmolLM2-360M-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32).to(device).eval()
        self.device = torch.device(device)
        self.max_input_tokens = 0

    def choice_probabilities(self, state, question, output_keys):
        options = '\n'.join(f'{k}: {v}' for k, v in question['criteria'].items())
        content = state + '\n' + question['instructions'] + '\n' + options
        content += '\nReply with one option letter only.'
        self.last_prompt = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(self.last_prompt, add_special_tokens=False, return_tensors='pt')
        prefix_ids = inputs['input_ids'][0].tolist()
        token_ids = []
        for key in output_keys:
            joined = self.tokenizer.encode(self.last_prompt + key, add_special_tokens=False)
            assert joined[:-1] == prefix_ids, 'Choice must be exactly one next token.'
            token_ids.append(joined[-1])
        assert len(set(token_ids)) == len(output_keys)
        self.last_input_tokens = len(prefix_ids)
        assert len(prefix_ids) <= self.model.config.max_position_embeddings, 'Context too long.'
        self.max_input_tokens = max(self.max_input_tokens, len(prefix_ids))
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
        assert torch.isfinite(logits).all()
        self.last_allowed_token_mass = float(torch.softmax(logits, -1)[token_ids].sum())
        self.last_choice_logits = logits[token_ids].tolist()
        p = torch.softmax(logits[token_ids], -1).double().cpu().numpy()
        return p / p.sum()


class QwenChoice(SmolLMChoice):
    """Same constrained typed-choice interface on cached Qwen2.5-1.5B-Instruct."""

    def __init__(self):
        super().__init__('Qwen/Qwen2.5-1.5B-Instruct')


class Qwen7BChoice(SmolLMChoice):
    """Higher-capacity contextual control on cached Qwen2.5-7B-Instruct."""

    def __init__(self):
        super().__init__('Qwen/Qwen2.5-7B-Instruct')
