"""Laya noul event reporting with unchanged default context and FP32 precision."""


def create_agent(device):
    from src.event_models import LayaEventProbability
    return LayaEventProbability(device)
