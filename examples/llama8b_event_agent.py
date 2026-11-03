"""Local Llama 3.1 8B Instruct, original two-token numerical report budget."""


def create_agent(device):
    from src.hf_choice_models import ROOT
    from src.numeric_event_models import FrozenNumericEvent
    return FrozenNumericEvent(ROOT / 'models/llama-3.1-8b-instruct', device, 'instruct_chat')
