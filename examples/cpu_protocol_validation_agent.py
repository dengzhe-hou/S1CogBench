"""CPU-only plumbing reference. This is not a fitted cognitive or LLM baseline.

Only visible state/question/ordered legal keys enter the policy. It uses no
scoring target, source answer, future feedback, model weight or network call.
"""

class CPUProtocolReference:
    def __init__(self):
        self.metadata = {
            'purpose': 'full-source CPU integration validation only; not a research result',
            'policy': 'visible-state-length modulo legal-action count; 0.75 on that key',
            'weights': 'none', 'training': 'none', 'device': 'cpu',
            'numeric_event_report': 'fixed explicit 50 percent; not action confidence',
            'restless_confidence': 'fixed independent 0.5 report through original parser',
        }

    def choice_probabilities(self, state, question, output_keys):
        if not output_keys or len(set(output_keys)) != len(output_keys):
            raise ValueError('Distinct original legal keys are required.')
        if set(question['criteria']) != set(output_keys):
            raise ValueError('Question and original legal keys disagree.')
        index = len(state) % len(output_keys)
        self.last_native_choice = output_keys[index]
        if len(output_keys) == 1:
            return [1.0]
        probability = [0.25 / (len(output_keys) - 1)] * len(output_keys)
        probability[index] = 0.75
        return probability

    def confidence_probability(self, state, chosen, original_specifications):
        return 0.5

    def posterior_probability(self, state, question):
        # The fixed explicit 50 percent event report in metadata. The validation
        # driver supplied it inline; the unified CLI asks the adapter.
        return 0.5


def create_agent(device):
    if str(device) != 'cpu':
        raise ValueError('This protocol-validation reference is CPU-only.')
    return CPUProtocolReference()