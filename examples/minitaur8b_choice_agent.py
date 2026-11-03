"""Cached current Minitaur revision; frozen BF16, recommended response markers."""


def create_agent(device):
    from src.hf_choice_models import ROOT, FrozenDecoderChoice
    # Reuse the exact checkpoint already evaluated, not a potentially moving HF alias.
    import os
    model_dir = os.environ.get('S1COGBENCH_MINITAUR_MODEL_DIR', str(ROOT / 'models/minitaur8b'))
    return FrozenDecoderChoice(model_dir, device, 'centaur_response_markers')
