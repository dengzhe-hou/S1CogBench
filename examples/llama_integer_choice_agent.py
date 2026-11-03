from pathlib import Path
from src.integer_label_choice import IntegerLabelChoice


def create_agent(device):
    return IntegerLabelChoice(Path(__file__).resolve().parents[1] / 'models/llama-3.1-8b-instruct',
                              device, 'instruct_chat')
