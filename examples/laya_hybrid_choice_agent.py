"""Frozen Laya choice head for the Hybrid task's current-statistics text interface."""
from examples.laya_bandit_choice_agent import create_agent as load_full_context_laya


def create_agent(device):
    agent = load_full_context_laya(device)
    agent.metadata['task_interface'] = ('Hybrid original current-decision X statistics rendered as text; '
                                        'four absolute sample/select actions. This is not the original stimulus.')
    return agent
