"""Laya's frozen SDK choice head with a full-input runtime budget for Bandit SH."""
from pathlib import Path

import numpy as np
import torch

from src.choice_models import LayaChoice


class BanditLayaChoice(LayaChoice):
    def __init__(self, device):
        super().__init__(device=device)
        self.agent.dtype = torch.float32
        self.sdk_default_max_len = self.agent.cfg['max_len']
        self.encoder_max_positions = int(self.agent.model.encoder.config.max_position_embeddings)
        self.agent.cfg['max_len'] = self.encoder_max_positions
        self.metadata = {
            'model': 'Laya English frozen local checkpoint',
            'checkpoint': str(Path(__file__).resolve().parents[1] / 'models/laya-english'),
            'sdk_default_max_len': self.sdk_default_max_len,
            'runtime_max_len': self.encoder_max_positions,
            'head_max_len': self.agent.cfg['head_max_len'],
            'forward_dtype': str(self.agent.dtype),
            'actual_parameter_dtypes': sorted({str(p.dtype) for p in self.agent.model.parameters()}),
            'training': 'No task fitting, calibration, or weight changes.',
            'input_processing': 'Full state tokenized before forward; raise if model context would truncate it.',
            'typed_choice': 'Official SDK choice head and official action-name order.',
            'temperature_by_options_raw': self.agent.temperature_by_options_raw,
            'temperature_by_options_effective': self.agent.temperature_by_options,
            'temperature_note': 'SDK clamps the checkpoint choice:11+ value 0.1005828 to 0.5; '
                                'retain this actual runtime behavior and report the affected confidence as uncalibrated.',
        }

    def choice_probabilities(self, state, question, output_keys):
        internal = {'t': 'choice', 'ins': question['instructions'], 'crit': question['criteria']}
        ids, markers = self.build_sequence(self.agent.tok, state, internal,
                                           100000, self.agent.cfg['head_max_len'])
        if len(markers) != len(output_keys):
            raise ValueError('The official Laya head budget cannot represent every legal Bandit action.')
        if len(ids) > self.encoder_max_positions:
            raise ValueError(f'Full Bandit input has {len(ids)} tokens, above this Laya encoder limit '
                             f'{self.encoder_max_positions}; no state history was truncated.')
        self.last_input_tokens = len(ids)
        self.max_input_tokens = max(self.max_input_tokens, len(ids))
        answer = self.agent.predict(state, {'action': question})['answers']['action']
        if self.agent.device != self.requested_device:
            raise RuntimeError(f'Laya changed device to {self.agent.device}; timing is not comparable.')
        raw = np.asarray([answer['probabilities'][key] for key in output_keys], dtype=float)
        if (not np.isfinite(raw).all() or (raw < 0).any() or raw.sum() <= 0
                or abs(raw.sum() - 1) > len(raw) * 0.00005 + 1e-8):
            raise ValueError('Laya SDK returned invalid rounded legal-choice probabilities.')
        self.last_raw_probabilities = raw.tolist()
        self.last_native_choice = answer['choice']
        return raw / raw.sum()


def create_agent(device):
    torch.set_num_threads(4)
    return BanditLayaChoice(device)
