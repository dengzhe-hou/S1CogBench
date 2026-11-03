"""Exp1 [4,1,2] task using the public Python Mouselab environment.

Frontier rule follows the original Julia allowed/has_observed_parent functions.
Numeric click cost is a declared diagnostic condition, not human 3-second delay.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/mouselab-mdp-tools'))
from mouselab.mouselab import MouselabEnv
from mouselab.distributions import Categorical

VALUES = (-10, -5, 5, 10)


class Exp1Env(MouselabEnv):
    def actions(self, state):
        if state is self.term_state:
            return
        yield self.term_action
        for parent, children in enumerate(self.tree):
            if not hasattr(state[parent], 'sample'):
                for child in children:
                    if hasattr(state[child], 'sample'):
                        yield child

    def step(self, action):
        assert action in list(self.actions(self._state)), 'Illegal reveal or terminal action.'
        return super().step(action)


def new_game(seed, cost):
    assert np.isfinite(cost) and cost >= 0
    tree, init = Exp1Env.branching_and_reward_to_inputs([4, 1, 2], Categorical(VALUES))
    rng = np.random.default_rng(seed)
    truth = np.concatenate(([0], rng.choice(VALUES, 16))).astype(float)
    env = Exp1Env(tree, init, ground_truth=truth, cost=cost,
                  term_belief=True, sample_term_reward=False, seed=seed)
    return env


def visible_belief(state):
    return [None if hasattr(value, 'sample') else float(value) for value in state]


def visible_text(env, reveals, cost):
    belief = visible_belief(env._state)
    edges = '; '.join(f'{i}->{",".join(map(str, children))}'
                      for i, children in enumerate(env.tree) if children)
    values = ', '.join(f'{i}={"?" if v is None else f"{v:g}"}' for i, v in enumerate(belief))
    history = ', '.join(f'{node}:{reward:g}' for node, reward in reveals) or 'none'
    return (
        'Choose which hidden node value to reveal, or STOP.\n'
        'Tree edges: ' + edges + '. Start at node 0 (value 0).\n'
        'Each other node independently has value -10, -5, 5, or 10, equally likely. '
        'A reveal gives its exact value. Reveal only an unseen node whose parent is known.\n'
        f'Each reveal costs {cost:g} points. STOP is free. On STOP the executor chooses '
        'the root-to-leaf path with greatest expected sum using known values and mean 0 '
        'for unknown values. Your payoff is that expected path sum minus all reveal costs. '
        'Maximize this payoff. You do not need to select the path.\n'
        'Node values (? = hidden): ' + values + '.\n'
        'Reveals in order (node:value): ' + history + '.\n'
        f'Reveal cost spent so far: {cost * len(reveals):g} points.'
    )


def choice_question(actions, term_action):
    keys = [chr(ord('A') + i) for i in range(len(actions))]
    question = {'type': 'choice', 'instructions': 'Choose the next action.',
                'criteria': {key: ('STOP now.' if action == term_action else f'Reveal node {action}.')
                             for key, action in zip(keys, actions)}}
    return question, keys


def one_step_net_voi(env, actions, cost):
    return np.array([0.0 if a == env.term_action else float(env.myopic_voc(a, env._state)) - cost
                     for a in actions])


def classic_probabilities(policy, env, actions, cost):
    if policy == 'random':
        return np.full(len(actions), 1 / len(actions))
    if policy == 'myopic_voi':
        gains = one_step_net_voi(env, actions, cost)
        if gains.max() <= 0:
            return np.array([float(a == env.term_action) for a in actions])
        tied = gains == gains.max()
        return tied.astype(float) / tied.sum()
    raise ValueError(policy)
