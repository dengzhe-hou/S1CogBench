"""Frozen local Laya English using the same FP32 arithmetic as existing runs."""


def create_agent(device):
    import torch
    from src.choice_models import LayaChoice

    torch.set_num_threads(4)
    agent = LayaChoice(device=device)
    agent.agent.dtype = torch.float32
    return agent
