class RandomAgent:
    metadata = {'model': 'Original random choice/confidence rules',
                'implementation': 'Coin toss F/J and discrete uniform confidence .01 through .99; '
                                  'separate callback RNG is a new replay adaptation.'}

    def choice_probabilities(self, state, question, keys):
        import numpy as np
        self.last_native_choice = keys[0] if np.random.rand() < .5 else keys[1]
        return [.5, .5]

    def confidence_probability(self, prompt, chosen, specifications):
        import numpy as np
        return np.random.randint(1, 100) / 100


def create_agent(device):
    if device != 'cpu':
        raise ValueError('Random reference is CPU-only.')
    return RandomAgent()
