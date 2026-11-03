"""Paper-form GCM; explicit reimplementation, not the missing official GCM code.

Only beta is fitted, jointly across original training participants. The fixed
kernel is exp(-Euclidean distance); memory contains previously revealed true
categories, never the current label. See docs/GCM_IMPLEMENTATION_20261017.md.
"""
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.psych_typed import DEFAULT_SOURCE, load_shj_trials
from src.psych_category_tasks import load_category_trials

TRAIN_SOURCE = Path(__file__).resolve().parents[1] / 'data/psych101/prompts_training.jsonl'
PROTOCOL = 'paper-form-gcm-unit-features-schedulefree126-v1'


@dataclass
class EvidenceRow:
    trial: object
    features: tuple
    evidence: tuple | None
    observed_examples: int
    status: str


def features_for(trial, task):
    if task == 'shj':
        match = re.search(r'You see a (big|small) (black|white) (square|triangle)\. You press $', trial.state)
        if match is None:
            raise ValueError('Unrecognized original SHJ stimulus.')
        return tuple(float(v == positive) for v, positive in zip(match.groups(), ('big', 'black', 'square')))
    if task == 'medin':
        if re.fullmatch(r'[12]{3}', trial.stimulus) is None:
            raise ValueError('Medin must have all three original binary dimensions.')
        return tuple(float(int(v) - 1) for v in trial.stimulus)
    if task == 'weather':
        cards = [int(v) for v in re.findall(r'card ([1-4])', trial.stimulus)]
        if not cards or len(cards) != len(set(cards)):
            raise ValueError('Weather stimulus must be a set of original cards.')
        return tuple(float(i in cards) for i in range(1, 5))
    raise ValueError(task)


def load_trials(task, source):
    return load_shj_trials(source) if task == 'shj' else load_category_trials(task, source)


def build_evidence(trials, task):
    """Causal sufficient statistics; each example is added after its prediction.

    SHJ memory resets on explicit new problem. Other tasks reset per session.
    Medin rating rows remain present and unsupported; no-feedback test choices
    are scored against the unchanged memory of revealed learning examples.
    """
    rows, memories, key_orders = [], {}, {}
    for trial in trials:
        x = features_for(trial, task)
        memory_id = (trial.session_id, trial.problem_index if task == 'shj' else None)
        memory = memories.setdefault(memory_id, Counter())
        n_seen = sum(memory.values())
        if task == 'medin' and trial.response_kind == 'typicality_rating':
            rows.append(EvidenceRow(trial, x, None, n_seen, 'unsupported_ordinal_typicality'))
            continue
        if key_orders.setdefault(memory_id, trial.output_keys) != trial.output_keys:
            raise ValueError('Category key order changed within an exemplar memory.')
        evidence = [0.0] * len(trial.output_keys)
        for (past_x, label), count in memory.items():
            similarity = math.exp(-math.sqrt(sum((a - b) ** 2 for a, b in zip(x, past_x))))
            evidence[trial.output_keys.index(label)] += count * similarity
        rows.append(EvidenceRow(trial, x, tuple(evidence), n_seen, 'scored_category'))
        if trial.correct_category is not None:
            memory[(x, trial.correct_category)] += 1
    return rows


def tensors(rows):
    supported = [row for row in rows if row.evidence is not None]
    evidence = torch.tensor([row.evidence for row in supported], dtype=torch.float64)
    choices = torch.tensor([row.trial.output_keys.index(row.trial.original_choice_label)
                            for row in supported], dtype=torch.long)
    return evidence, choices


class GCM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Main official GCM init/constraints are unavailable. This direct real
        # beta, deterministic zero init and float64 are disclosed choices.
        self.beta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, evidence):
        return self.beta * evidence


def fit_gcm(evidence, choices, iterations=1000, lr=0.1, progress=None):
    from schedulefree import AdamWScheduleFree
    model = GCM()
    optimizer = AdamWScheduleFree(model.parameters(), lr=lr)
    model.train()
    optimizer.train()
    trace = []
    for step in range(iterations):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(evidence), choices)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training likelihood at iteration {step + 1}.')
        loss.backward()
        optimizer.step()
        row = {'iteration': step + 1, 'train_nll_before_update': float(loss.detach()),
               'beta_train_mode_after_update': float(model.beta.detach())}
        trace.append(row)
        if progress is not None:
            progress(row)
    model.eval()
    optimizer.eval()  # Required by the actual official schedule-free Trainer.
    with torch.no_grad():
        final_nll = float(F.cross_entropy(model(evidence), choices))
    return model, trace, final_nll


def score_rows(rows, model):
    supported = [row for row in rows if row.evidence is not None]
    evidence, _ = tensors(rows)
    with torch.no_grad():
        logs = F.log_softmax(model(evidence), dim=1).tolist()
    scored = iter(zip(supported, logs))
    records = []
    for row in rows:
        record = row.trial.identifiers()
        record.update({'features': list(row.features), 'previous_revealed_examples': row.observed_examples,
                       'status': row.status, 'protocol': PROTOCOL})
        if row.evidence is None:
            record.update({'probabilities': None, 'human_choice_nll': None,
                           'reason': 'Published categorical GCM has no ordinal typicality readout.'})
        else:
            scored_row, log_probs = next(scored)
            assert scored_row is row
            keys = row.trial.output_keys
            probabilities = [math.exp(v) for v in log_probs]
            choice = keys[max(range(len(log_probs)), key=log_probs.__getitem__)]
            target = keys.index(row.trial.original_choice_label)
            record.update({'category_evidence': list(row.evidence),
                           'probabilities': dict(zip(keys, probabilities)),
                           'log_probabilities': dict(zip(keys, log_probs)),
                           'human_choice_nll': -log_probs[target],
                           'human_choice_probability_underflow': probabilities[target] == 0.0,
                           'choice': choice, 'choice_readout': 'argmax, first original key on exact tie',
                           'human_choice_agreement': choice == row.trial.original_choice_label,
                           'objective_category_accuracy': (choice == row.trial.correct_category
                                                           if row.trial.correct_category is not None else None),
                           'invalid_response': False})
        records.append(record)
    return records


def inventory(trials):
    participants = {(t.experiment, str(t.participant)) for t in trials}
    return {'responses': len(trials), 'sessions': len({t.session_id for t in trials}),
            'experiment_participants': len(participants),
            'participants': [{'experiment': experiment, 'participant': participant}
                             for experiment, participant in sorted(participants)],
            'response_kinds': dict(Counter(getattr(t, 'response_kind', 'category_choice') for t in trials)),
            'experiments': dict(Counter(t.experiment for t in trials))}


def summarize(records):
    scored = [r for r in records if r['status'] == 'scored_category']
    objective = [r['objective_category_accuracy'] for r in scored
                 if r['objective_category_accuracy'] is not None]
    return {'source_responses': len(records), 'scored_category_responses': len(scored),
            'unsupported_responses': len(records) - len(scored),
            'human_choice_mean_nll': float(np.mean([r['human_choice_nll'] for r in scored])),
            'human_choice_agreement': float(np.mean([r['human_choice_agreement'] for r in scored])),
            'objective_category_accuracy': float(np.mean(objective)) if objective else None,
            'objective_responses': len(objective),
            'invalid_response_count': sum(r['invalid_response'] for r in scored),
            'human_choice_probability_underflow_count': sum(r['human_choice_probability_underflow'] for r in scored)}
