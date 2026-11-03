"""Complete integer replies for Qwen3.5; separate from next-letter readouts.

Each event is the literal integer followed immediately by <|im_end|>.  Scores
include every token and the terminator, without length normalization.  This is
an explicit benchmark readout, not the archived original full-vocabulary NLL.
"""
from copy import deepcopy

from examples.qwen35_4b_choice_agent import Qwen35Choice

TURN_END = "<|im_end|>"
ASSISTANT_START = "<|im_start|>"
INTEGER_PREPARATION_IMPLEMENTATION = "packed-candidate-ids-length-reuse-v1-20261019"


def is_integer_question(output_keys):
    return bool(output_keys) and all(isinstance(key, str) and key.isascii() and key.isdigit()
                                     for key in output_keys)


def format_integer_prompt(tokenizer, state, question, output_keys):
    if not is_integer_question(output_keys) or len(set(output_keys)) != len(output_keys):
        raise ValueError("Supply distinct original integer report strings.")
    if set(output_keys) != set(question["criteria"]):
        raise ValueError("Question criteria and legal reports differ.")
    content = state + "\n" + question["instructions"]
    content += "\nLegal integer reports: " + ", ".join(output_keys) + "."
    content += "\nReturn only one legal integer."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def integer_continuations(tokenizer, prompt, prefix, output_keys):
    """Verify the actual answer boundary without retokenizing history 101 times.

    Qwen's final assistant segment begins at an atomic added special token.
    That boundary separates it from all user text.  Retokenize this exact short
    segment plus each complete reply, and check it against the full prompt's
    suffix.  CPU preparation also checks full concatenations on real inputs.
    """
    start_id = tokenizer.convert_tokens_to_ids(ASSISTANT_START)
    end_id = tokenizer.convert_tokens_to_ids(TURN_END)
    added = tokenizer.get_added_vocab()
    if (added.get(ASSISTANT_START) != start_id or added.get(TURN_END) != end_id
            or tokenizer.encode(TURN_END, add_special_tokens=False) != [end_id]):
        raise ValueError("Expected Qwen atomic chat boundary tokens.")
    boundary = prompt.rfind(ASSISTANT_START)
    if boundary < 0 or not prompt[boundary:].startswith(ASSISTANT_START + "assistant\n"):
        raise ValueError("Expected the official final assistant generation prefix.")
    suffix = prompt[boundary:]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if suffix_ids[0] != start_id or prefix[-len(suffix_ids):] != suffix_ids:
        raise ValueError("Actual assistant boundary tokenization differs from the full prompt.")
    sequences = []
    for key in output_keys:
        joined = tokenizer.encode(suffix + key + TURN_END, add_special_tokens=False)
        if joined[:len(suffix_ids)] != suffix_ids:
            raise ValueError("Integer reply changes the actual prompt tokenization.")
        sequence = joined[len(suffix_ids):]
        if len(sequence) < 2 or sequence[-1] != end_id:
            raise ValueError("Each integer reply must include its complete body and turn terminator.")
        sequences.append(sequence)
    if len({tuple(sequence) for sequence in sequences}) != len(output_keys):
        raise ValueError("Integer replies must define distinct complete token sequences.")
    return sequences


def score_integer_sequences(model, inputs, sequences):
    """One full prefill, then one isolated hybrid-cache continuation per reply.

    Qwen's convolution and recurrent states mutate in place.  Deep-copy the
    complete DynamicCache, including those states and attention K/V.  Cropping
    a KV cache or a shallow copy cannot restore an independent Qwen branch.
    Only the prompt cache and one working branch are retained at a time.
    Pack candidate IDs once and reuse mask/position tensors by body length;
    all model calls retain their original order and single-reply shapes.
    """
    import torch

    with torch.inference_mode():
        packed = torch.tensor([token for sequence in sequences for token in sequence],
                              device=inputs["input_ids"].device)
        prefix_length = inputs["input_ids"].shape[1]
        by_length = {}
        prepared = []
        offset = 0
        for sequence in sequences:
            # The final terminator is scored from the last integer token's
            # logits, so there is no need to feed the terminator itself.
            length = len(sequence) - 1
            continuation = packed[offset:offset + length].unsqueeze(0)
            targets = packed[offset + 1:offset + len(sequence)]
            if length not in by_length:
                attention_mask = torch.cat([
                    inputs["attention_mask"], torch.ones_like(continuation)
                ], dim=-1)
                position_ids = torch.arange(prefix_length, prefix_length + length,
                                            device=continuation.device).unsqueeze(0)
                by_length[length] = attention_mask, position_ids
            attention_mask, position_ids = by_length[length]
            prepared.append((continuation, targets, attention_mask, position_ids))
            offset += len(sequence)

        prefill = model(**inputs, use_cache=True, logits_to_keep=1)
        cache = prefill.past_key_values
        if cache is None:
            raise ValueError("Qwen must return its full hybrid cache for integer scoring.")
        first_log_probs = prefill.logits[0, -1].float().log_softmax(-1)
        del prefill
        scores = []
        for sequence, (continuation, targets, attention_mask, position_ids) in zip(sequences, prepared):
            branch = deepcopy(cache)
            output = model(input_ids=continuation, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=branch,
                           use_cache=True, logits_to_keep=0)
            log_probs = output.logits[0].float().log_softmax(-1)
            suffix_score = log_probs.gather(1, targets[:, None]).squeeze(1).double().sum()
            scores.append(first_log_probs[sequence[0]].double() + suffix_score)
            del output, branch, log_probs
        scores = torch.stack(scores)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite complete integer reply log probabilities.")
        return scores


class Qwen35IntegerSequenceChoice(Qwen35Choice):
    def __init__(self, device):
        super().__init__(device)
        self.metadata.update(
            interface="Complete integer sequence readout with explicit turn termination; official non-thinking chat template. Letter questions retain Qwen35Choice.",
            integer_interface="All original integer strings, with every token scored; no bins, letter remapping or digit subquestions.",
            probabilities="Integer: temperature-one joint sequence log probabilities conditioned on the complete legal reply set. Letter: existing legal next-token softmax.",
            integer_termination={"text": TURN_END,
                                 "token_id": self.tokenizer.convert_tokens_to_ids(TURN_END),
                                 "immediate": True,
                                 "scored": True},
            integer_event="P(integer tokens followed immediately by <|im_end|> | complete chat prompt). No trailing whitespace or alternative termination is included.",
            integer_conditioning="exp(log P(reply) - logsumexp over all legal complete replies); no length normalization.",
            integer_cache="One full prefill per response; sequential complete DynamicCache deep copies isolate attention, convolution and recurrent states; no cross-response cache reuse.",
            integer_preparation_implementation=INTEGER_PREPARATION_IMPLEMENTATION,
            integer_cost="One prefill plus one short continuation and full cache copy per legal integer (up to 101); GPU time and memory require actual sanity measurement.",
            raw_probabilities="On integer rows, unconditioned complete terminated-reply probabilities in original action order; on letter rows, absent.",
            prompt_relation="Integer wording matches the typed full-integer prompt; non-thinking Qwen chat and explicit terminated-sequence scoring differ from Llama-family one-token and archived author readouts.",
            nll_scope="Conditional legal-action NLL from the stated readout; not archived original full-vocabulary NLL or an official matched inference protocol.",
            readout_temperature=1,
        )

    def choice_probabilities(self, state, question, output_keys):
        if not is_integer_question(output_keys):
            # Self-assessment alternates full-integer reports with letter items.
            # Avoid leaving the preceding integer row's auxiliary values here.
            for name in ("last_raw_probabilities", "last_sequence_logprobabilities",
                         "last_allowed_sequence_mass", "last_max_reply_tokens"):
                self.__dict__.pop(name, None)
            return super().choice_probabilities(state, question, output_keys)

        import numpy as np
        import torch

        prompt = format_integer_prompt(self.tokenizer, state, question, output_keys)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prefix = inputs["input_ids"][0].tolist()
        sequences = integer_continuations(self.tokenizer, prompt, prefix, output_keys)
        self.last_input_tokens = len(prefix)
        self.last_max_reply_tokens = max(map(len, sequences))
        if self.last_input_tokens + self.last_max_reply_tokens > self.max_context_tokens:
            raise ValueError("Complete history plus complete reply exceeds context; truncation is forbidden.")
        self.max_input_tokens = max(self.max_input_tokens, self.last_input_tokens)
        self.last_prompt = prompt
        self.last_label_mapping = dict(zip(output_keys, output_keys))
        self.__dict__.pop("last_allowed_token_mass", None)
        scores = score_integer_sequences(self.model, inputs.to(self.device), sequences)
        probability = torch.softmax(scores, dim=-1).cpu().numpy()
        self.last_sequence_logprobabilities = scores.cpu().tolist()
        self.last_raw_probabilities = scores.exp().cpu().tolist()
        self.last_allowed_sequence_mass = float(torch.logsumexp(scores, dim=0).exp())
        self.last_native_choice = output_keys[int(np.argmax(probability))]
        return probability


def create_agent(device):
    return Qwen35IntegerSequenceChoice(device)
