from pathlib import Path
from src.integer_label_choice import IntegerLabelChoice


def create_agent(device):
    return IntegerLabelChoice(Path('models/minitaur8b'),
                              device, 'centaur_response_markers')
