"""Compact risky-choice search presentation, not used by the evaluated modes."""
from src.psych_long_history_compact import load_trials as _load
from src.psych_typed import DEFAULT_SOURCE


def load_trials(source=DEFAULT_SOURCE):
    return _load('risky_choice_information_search', source)
