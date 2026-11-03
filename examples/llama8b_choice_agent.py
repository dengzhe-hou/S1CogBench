"""Local Llama 3.1 8B Instruct; frozen BF16 typed-choice comparison."""


def create_agent(device):
    from src.hf_choice_models import ROOT, FrozenDecoderChoice
    return FrozenDecoderChoice(ROOT / 'models/llama-3.1-8b-instruct', device, 'instruct_chat')
