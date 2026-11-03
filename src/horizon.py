"""A small adapter to the unmodified public CogBench Horizon environment.

The environment generates synthetic experimental trials, not human observations.
Only chosen rewards enter the policy's history. Latent means are scoring-only.
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from choice_models import LayaChoice

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/CogBench/Experiments/HorizonTask/envs/bandits.py'
spec = importlib.util.spec_from_file_location('cogbench_horizon', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def new_game(seed, horizon=None, unequal=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = module.HorizonTaskWilson(
        num_actions=2, reward_scaling=1, reward_std=8, num_forced_choice=4)
    env.reset(unequal=None if unequal is None else np.array([unequal]), horizon=horizon)
    history = [(int(env.action[t, 0]), float(env.rewards[t, 0, env.action[t, 0]]))
               for t in range(4)]
    return env, history


def visible_text(history, rounds_left, labels=('F', 'J'), summary=False):
    # Complete chronological history; no hidden means, unchosen rewards, or
    # derived posterior are sent to any model.
    intro = ('You are going to a casino that owns two slot machines.\n'
             'You earn money each time you play on one of these machines.\n'
             'You have received the following amount of dollars when playing in the past:\n')
    past = ''.join(f'- Machine {labels[a]} delivered {r:g} dollars.\n' for a, r in history)
    state = intro + past + f'Your goal is to maximize the sum of received dollars within {rounds_left} additional rounds.'
    if summary:
        state += '\nSummary computed only from the observations above:\n'
        for a, label in enumerate(labels):
            rewards = [r for arm, r in history if arm == a]
            state += f'- Machine {label}: {len(rewards)} observations; average {np.mean(rewards):.6g} dollars.\n'
    return state


def choice_question(labels=('F', 'J'), option_order=(0, 1)):
    return {'type': 'choice', 'instructions': 'Which machine do you choose?',
            'criteria': {labels[a]: f'Play machine {labels[a]}.' for a in option_order}}


def classic_probabilities(name, history):
    if name == 'random':
        return np.array([0.5, 0.5])
    if name == 'empirical_greedy':
        means = np.array([np.mean([r for a, r in history if a == k]) for k in range(2)])
        tied = means == means.max()
        return tied.astype(float) / tied.sum()
    raise ValueError(f'Unknown policy: {name}')


def check_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    assert p.shape == (2,) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(), 1, atol=1e-6, rtol=0)
    return p


class LayaPolicy(LayaChoice):
    def probabilities(self, history, rounds_left, labels=('F', 'J'), option_order=(0, 1), summary=False):
        state = visible_text(history, rounds_left, labels, summary)
        question = choice_question(labels, option_order)
        return check_probabilities(self.choice_probabilities(state, question, labels))
