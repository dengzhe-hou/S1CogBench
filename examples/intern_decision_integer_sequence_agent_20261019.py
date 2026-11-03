"""User-approved additional Intern integer readout, 2026-10-19.

All original numeric reports use complete integer + immediate turn termination
at T=1. Ordinary choices retain the official frozen BF16 Intern interface.
This additional protocol does not change the recorded native 62-symbol limit.
"""
from examples.intern_decision_choice_agent import InternDecisionChoice
from examples.qwen35_4b_integer_sequence_agent import (
    INTEGER_PREPARATION_IMPLEMENTATION, TURN_END, format_integer_prompt, integer_continuations,
    is_integer_question, score_integer_sequences,
)

PROTOCOL = "intern-approved-complete-integer-sequence-20261019"


class InternDecisionIntegerSequenceChoice(InternDecisionChoice):
    def __init__(self, device):
        super().__init__(device)
        self.metadata.update(
            readout_protocol=PROTOCOL,
            protocol_status="User explicitly approved option A, full integer readout, 2026-10-19.",
            interface="Additional integer sequence readout; ordinary choices retain the official native interface.",
            probabilities="Numeric: complete terminated-reply joint probability conditioned on all original legal values, T=1. Ordinary choice: official calibrated symbol distribution.",
            numeric_interface="Literal original integer followed immediately by <|im_end|>; every token scored; no length normalization, bins, or digit subquestions.",
            numeric_prompt="Official non-thinking chat template with complete original human history and legal integer reports.",
            numeric_temperature=1,
            numeric_termination={"text": TURN_END, "token_id": self.engine.tokenizer.convert_tokens_to_ids(TURN_END), "scored": True},
            numeric_calibration="Official 1.99241824 symbol calibration is not transferred to complete sequence likelihood.",
            numeric_cost="One full prefill plus one independent full hybrid-cache continuation per legal reply; measured separately from native inference.",
            integer_preparation_implementation=INTEGER_PREPARATION_IMPLEMENTATION,
            native_choice="Numeric: argmax of legal complete-sequence probabilities, first legal key on a tie. Ordinary choice: official native choice and tie-breaking.",
            letter_interface="Unmodified official InternDecisionChoice, checkpoint symbol mapping and calibration.",
            interpretation="Additional frozen-checkpoint protocol, not official native 101-way capability or archived full-vocabulary NLL.",
        )

    def choice_probabilities(self, state, question, output_keys):
        if not is_integer_question(output_keys):
            for name in ("last_sequence_logprobabilities", "last_allowed_sequence_mass",
                         "last_max_reply_tokens", "last_integer_token_sequences"):
                self.__dict__.pop(name, None)
            self.last_readout_interface = "official_native_symbol"
            return super().choice_probabilities(state, question, output_keys)

        import numpy as np
        import torch

        backend, tokenizer = self.engine.backend, self.engine.tokenizer
        prompt = format_integer_prompt(tokenizer, state, question, output_keys)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prefix = inputs["input_ids"][0].tolist()
        sequences = integer_continuations(tokenizer, prompt, prefix, output_keys)
        self.last_input_tokens = len(prefix)
        self.last_max_reply_tokens = max(map(len, sequences))
        if len(prefix) + self.last_max_reply_tokens > backend.max_length:
            raise ValueError("Complete history plus complete reply exceeds the unchanged 8192-token budget.")
        self.max_input_tokens = max(self.max_input_tokens, len(prefix))
        self.last_prompt = prompt
        self.last_label_mapping = dict(zip(output_keys, output_keys))
        self.last_integer_token_sequences = sequences
        self.last_readout_interface = "complete_integer_sequence_t1"
        for name in ("last_allowed_token_mass", "last_response"):
            self.__dict__.pop(name, None)
        scores = score_integer_sequences(backend.model, inputs.to(backend.device), sequences)
        probability = torch.softmax(scores, dim=-1).cpu().numpy()
        self.last_sequence_logprobabilities = scores.cpu().tolist()
        self.last_raw_probabilities = scores.exp().cpu().tolist()
        self.last_allowed_sequence_mass = float(torch.logsumexp(scores, dim=0).exp())
        self.last_native_choice = output_keys[int(np.argmax(probability))]
        return probability


def create_agent(device):
    return InternDecisionIntegerSequenceChoice(device)
