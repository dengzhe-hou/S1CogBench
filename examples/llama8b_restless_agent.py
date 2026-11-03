def create_agent(device):
    from examples.llama8b_choice_agent import create_agent as load_choice
    from src.restless_models import DecoderRestless
    return DecoderRestless(load_choice(device))
