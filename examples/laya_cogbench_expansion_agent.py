"""Reuse frozen Laya's existing full-input adapter for the new CogBench tasks."""
from examples.laya_bandit_choice_agent import create_agent as create_full_input_agent


def create_agent(device):
    agent = create_full_input_agent(device)
    agent.metadata['typed_choice'] = 'Original CogBench visible state and ordered legal options; SDK native choice.'
    return agent
