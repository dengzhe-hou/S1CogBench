"""Qwentaur-14B-r64 frozen CogBench V1 numeric-report adaptation."""


def create_agent(device):
    from src.qwentaur_numeric_event_models import Qwen14BNumericEvent
    return Qwen14BNumericEvent(device, 'qwentaur14b-r64')
