def create_agent(device):
    from src.decision4b_event_models import Qwen35NumericEvent
    return Qwen35NumericEvent(device)
