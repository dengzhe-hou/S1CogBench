"""Local Llama 3.1 8B Instruct, the numeric report of the accepted nonliteral-number run."""
from types import SimpleNamespace


class SocialNumeric:
    def __init__(self, device):
        from src.social_numeric_models import NumericSocialModel
        self.model = NumericSocialModel('llama8b', device)
        self.metadata = self.model.metadata

    def posterior_probability(self, state, question):
        result = self.model.predict(SimpleNamespace(state=state, question=question))
        self.last_input_tokens, self.last_raw_answer = result['input_tokens'], result['raw_answer']
        return result['reported_probability']


def create_agent(device):
    return SocialNumeric(device)
