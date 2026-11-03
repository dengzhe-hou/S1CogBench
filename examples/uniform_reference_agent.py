"""Deterministic uniform reference for every adapter method. Not a research baseline.

choice_probabilities is uniform over the legal keys. last_native_choice is the
first legal key, the action the runners' documented argmax rule takes on a
uniform tie, so modes that require a native choice (WCST, the stationary
bandits) also run. posterior_probability and confidence_probability report
0.5. `python -m s1cogbench eval --smoke` uses this file; its stored expected
values were produced with it.

The Table 3 Uniform column is the expected score of uniform legal choice. For
human-NLL targets that equals this adapter's score; for accuracy, reward and
numeric targets it is an analytic expectation that one run of a fixed policy
does not reproduce (see docs/REPRODUCTION.md).
"""


class UniformReference:
    metadata = {'policy': 'uniform legal probabilities; native choice = first legal key; '
                          'event and confidence reports 0.5',
                'weights': 'none', 'training': 'none'}

    def choice_probabilities(self, state, question, output_keys):
        self.last_native_choice = output_keys[0]
        return [1.0 / len(output_keys)] * len(output_keys)

    def posterior_probability(self, state, question):
        return 0.5

    def confidence_probability(self, state, chosen, original_specifications):
        return 0.5


def create_agent(device):
    return UniformReference()
