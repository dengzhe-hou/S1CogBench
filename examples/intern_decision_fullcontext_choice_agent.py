"""Official Intern-Decision readout with its checkpoint's complete input budget.

The published engine's max_length is a rejection guard, not a token transform or
model configuration. Raise only that guard for new full-history jobs. Existing
8192-budget runs and their adapter file remain unchanged.
"""
from examples.intern_decision_choice_agent import InternDecisionChoice


def create_agent(device):
    agent = InternDecisionChoice(device)
    backend = agent.engine.backend
    default_limit = backend.max_length
    limit = int(backend.model.config.text_config.max_position_embeddings)
    backend.max_length = limit
    agent.metadata.update(
        max_input_tokens=limit,
        engine_default_max_length=default_limit,
        history='Full supplied history. Official tokenizer and encoder unchanged; only input rejection guard raised to checkpoint max positions.',
        input_budget_implementation='HFBackend.max_length is read only by encode() to reject over-budget input; weights, RoPE, temperature and output scoring unchanged.')
    return agent
