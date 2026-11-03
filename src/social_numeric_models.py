"""Explicit probability reports for CogGym numeric-event questions; never action confidence."""
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    'laya': ROOT / 'models/laya-english',
    'intern_decision_4b': ROOT / 'models/intern-decision-4b',
    'llama8b': ROOT / 'models/llama-3.1-8b-instruct',
    'minitaur8b': ROOT / 'models/minitaur8b',
    'qwen35_4b': ROOT / 'models/qwen3.5-4b',
}


def numeric_prompt(tokenizer, trial, model):
    content = trial.state + '\nReturn only the numerical percentage from 0 to 100 for the TARGET QUESTION. Do not add an explanation.'
    if model == 'minitaur8b':
        return content + '\nYou report <<'
    kwargs = {'enable_thinking': False} if model == 'qwen35_4b' else {}
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def parse_percentage(text):
    # Generation can run past the closing marker, so only the text before the marker is the report.
    match = re.fullmatch(r'((?:0|[1-9][0-9]*)(?:\.[0-9]+)?)\s*%?\s*(?:(?:>>|»).*)?', text.strip(), re.DOTALL)
    if not match:
        return None
    value = float(match.group(1))
    return value / 100 if 0 <= value <= 100 else None


def intern_request(trial):
    return {'state': trial.state, 'questions': {'event': trial.question}}


def audit_numeric_lengths(trials, models):
    from transformers import AutoTokenizer
    results = {}
    for name in models:
        path = DIRECTORIES[name]
        cfg = json.loads((path / ('encoder/config.json' if name == 'laya' else 'config.json')).read_text())
        limit = cfg.get('text_config', cfg)['max_position_embeddings']
        tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer' if name == 'laya' else path, local_files_only=True)
        lengths = []
        if name == 'laya':
            sys.path.insert(0, str(ROOT / 'external/laya'))
            from laya.common import build_sequence
            head = json.loads((path / 'rl_agent_config.json').read_text())['head_max_len']
        elif name == 'intern_decision_4b':
            from examples.intern_decision_choice_agent import load_official_inference
            official = load_official_inference()
            limit = 8192
        for trial in trials:
            if name == 'laya':
                q = {'t': 'noul', 'ins': trial.question['instructions']}
                ids, markers = build_sequence(tokenizer, trial.state, q, 100000, head)
                ref_ids, ref_markers = build_sequence(tokenizer, trial.state, q, 100000, 100000)
                if ids != ref_ids or markers != ref_markers:
                    raise ValueError(f'Laya question/criterion truncation: {trial.id}')
                n = len(ids)
            elif name == 'intern_decision_4b':
                compiled = official.compile_row(intern_request(trial))
                prompt = tokenizer.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False, enable_thinking=False, add_vision_id=True)
                n = len(tokenizer.encode(prompt, add_special_tokens=False))
            else:
                n = len(tokenizer.encode(numeric_prompt(tokenizer, trial, name), add_special_tokens=False))
            lengths.append(n)
        results[name] = {'responses_checked': len(trials), 'minimum_input_tokens': min(lengths),
            'maximum_input_tokens': max(lengths), 'median_input_tokens': statistics.median(lengths),
            'sum_input_tokens': sum(lengths), 'current_adapter_context_limit': limit,
            'responses_exceeding_current_context': sum(n > limit for n in lengths),
            'weights_loaded': False, 'truncation': False,
            'laya_head_tokens_match_unclipped_reference': True if name == 'laya' else None}
    return results


class NumericSocialModel:
    def __init__(self, model, device):
        import torch
        self.name, self.device = model, torch.device(device)
        if model == 'laya':
            from examples.laya_bandit_choice_agent import BanditLayaChoice
            self.base = BanditLayaChoice(device)
        elif model == 'intern_decision_4b':
            from examples.intern_decision_choice_agent import InternDecisionChoice
            self.base = InternDecisionChoice(device)
        elif model == 'qwen35_4b':
            from examples.qwen35_4b_choice_agent import Qwen35Choice
            self.base = Qwen35Choice(device)
        else:
            from src.hf_choice_models import FrozenDecoderChoice
            self.base = FrozenDecoderChoice(DIRECTORIES[model], device,
                'centaur_response_markers' if model == 'minitaur8b' else 'instruct_chat')
        self.metadata = {'backbone_loader_metadata': self.base.metadata, 'numeric_protocol':
            'Explicit event noul for Laya/Intern; greedy numerical percentage report for LLMs. These are not legal-action probabilities.',
            'max_new_tokens_llm': 16, 'do_sample_llm': False, 'invalid_output':
            'Retained as invalid with raw completion; no retries, probability conversion, or random fallback.',
            'source_slider': 'Original 0–100 percentage, converted to/from 0–1 only for storage.',
            'timing': 'Shared-device latency, not human reaction time.',
            'protocol_relation': 'Added per-query typed/numeric interface, not a rerun of archived CogGym Gemini results.'}

    @classmethod
    def from_loaded_choice(cls, model, base, device):
        """Reuse an already loaded choice model for a separate explicit event query."""
        import torch
        obj = object.__new__(cls)
        obj.name, obj.base, obj.device = model, base, torch.device(device)
        obj.metadata = {'reuses_loaded_choice_model': True, 'model': model,
                        'readout': 'Native noul for Intern/Laya; greedy explicit percentage for decoder controls.',
                        'max_new_tokens_llm': 16, 'invalid_completion_returns_none': True}
        return obj

    def predict(self, trial):
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.monotonic()
        if self.name == 'laya':
            agent = self.base.agent
            internal = agent._to_internal(trial.question)
            ids, markers = self.base.build_sequence(agent.tok, trial.state, internal, 100000, agent.cfg['head_max_len'])
            ref, ref_markers = self.base.build_sequence(agent.tok, trial.state, internal, 100000, 100000)
            if ids != ref or markers != ref_markers or len(ids) > self.base.encoder_max_positions:
                raise ValueError('Laya full event input would be truncated.')
            raw = agent.predict(trial.state, {'event': trial.question})['answers']['event']
            p, n = float(raw['noul']), len(ids)
        elif self.name == 'intern_decision_4b':
            raw = self.base.engine.predict(intern_request(trial))
            p, n = float(raw['answers']['event']['noul']), int(raw['usage']['input_tokens'])
        else:
            base = self.base
            prompt = numeric_prompt(base.tokenizer, trial, self.name)
            inputs = base.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            n = int(inputs.input_ids.shape[1])
            limit = base.max_context_tokens if self.name == 'qwen35_4b' else base.model.config.max_position_embeddings
            if n + 16 > limit:
                raise ValueError('Full numeric input exceeds context; no truncation.')
            with torch.inference_mode():
                generated = base.model.generate(**inputs.to(self.device), max_new_tokens=16, do_sample=False,
                    pad_token_id=base.tokenizer.eos_token_id)
            tokens = generated[0, n:].tolist()
            text = base.tokenizer.decode(tokens, skip_special_tokens=True)
            p = parse_percentage(text)
            raw = {'generated_text': text, 'generated_token_ids': tokens,
                   'generated_token_count': len(tokens), 'max_new_tokens': 16, 'do_sample': False}
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError('Native event interface returned an invalid probability.')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return {'reported_probability': p, 'raw_answer': raw, 'input_tokens': n,
                'batch_elapsed_seconds': time.monotonic() - start,
                'inference_interface': 'native_noul' if self.name in {'laya', 'intern_decision_4b'} else 'greedy_numeric_percentage'}
