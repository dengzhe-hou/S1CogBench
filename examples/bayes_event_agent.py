"""Original V1 visible-information Bayesian reference; no hidden scoring fields."""


def create_agent(device):
    from src.event_models import BayesEventProbability
    return BayesEventProbability()
