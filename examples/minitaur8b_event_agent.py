"""Current cached Minitaur revision, two-token numeric response with markers."""


def create_agent(device):
    from src.hf_choice_models import ROOT
    from src.numeric_event_models import FrozenNumericEvent
    import os
    model_dir = os.environ.get('S1COGBENCH_MINITAUR_MODEL_DIR', str(ROOT / 'models/minitaur8b'))
    return FrozenNumericEvent(model_dir, device, 'centaur_response_markers')
