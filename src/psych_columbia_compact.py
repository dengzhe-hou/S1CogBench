"""Prepared Columbia presentation; approval is pending for formal deployment."""
from src.psych_long_history_compact import load_trials as _load
from src.psych_typed import DEFAULT_SOURCE


def load_trials(source=DEFAULT_SOURCE):
    return _load('columbia_card_task', source)
