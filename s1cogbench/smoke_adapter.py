"""Adapter file used only by `eval --smoke` for runners that load adapters themselves:
the user's adapter, loaded once per worker, under the smoke call budget."""


def create_agent(device):
    from s1cogbench import smoke
    return smoke.budget_agent(device)
