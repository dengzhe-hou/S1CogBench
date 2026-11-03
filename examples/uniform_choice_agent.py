"""CPU-only interface reference; a new analytic policy, not the paper's fitted RW baseline."""


class UniformChoiceAgent:
    def choice_probabilities(self, state, question, output_keys):
        # The runner's sample rule samples this distribution. The native rule
        # uses its documented probability-argmax fallback because no native
        # choice is supplied; uniform ties therefore choose output_keys[0].
        return [1.0 / len(output_keys)] * len(output_keys)


def create_agent(device):
    if device != 'cpu':
        raise ValueError('The uniform interface reference is CPU-only.')
    return UniformChoiceAgent()
