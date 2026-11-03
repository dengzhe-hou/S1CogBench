"""Reuse A-Z encodings at the two fixed answer boundaries used here.

Qwen starts at its atomic assistant token. The decoder marker starts with
``You`` after a newline, which is a separate pre-token in the local tokenizer.
The complete prompt is still encoded once and its actual suffix checked on
every request. Only one short suffix is retained, never histories.
"""
import string

IMPLEMENTATION = 'fixed-answer-boundary-letter-tokens-v1-20261019'


class ChoiceLetterTokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.suffix = None

    def token_ids(self, prompt, prefix, labels, *, qwen=False):
        if qwen:
            marker = '<|im_start|>'
            start = prompt.rfind(marker)
            if (start < 0 or not prompt[start:].startswith(marker + 'assistant\n')
                    or self.tokenizer.get_added_vocab().get(marker)
                    != self.tokenizer.convert_tokens_to_ids(marker)):
                raise ValueError('Expected the atomic Qwen assistant boundary.')
            suffix = prompt[start:]
        else:
            suffix = 'You choose <<'
            if not prompt.endswith('\n' + suffix):
                raise ValueError('Expected the fixed decoder answer marker after a newline.')

        if suffix != self.suffix:
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            if not qwen:
                # Newline ends the preceding regex pre-token. Check the actual
                # local tokenizer, including its treatment of every A-Z suffix.
                pre = self.tokenizer.backend_tokenizer.pre_tokenizer
                parts = pre.pre_tokenize_str('\n' + suffix)
                tail = pre.pre_tokenize_str(suffix)
                if [(s, (a - 1, b - 1)) for s, (a, b) in parts[1:]] != tail:
                    raise ValueError('Decoder marker does not start at a pre-token boundary.')
                for label in string.ascii_uppercase:
                    joined_parts = pre.pre_tokenize_str(suffix + label)
                    if joined_parts[:-1] != tail or joined_parts[-1][1] != (len(suffix), len(suffix) + 1):
                        raise ValueError('Letter changes the decoder marker pre-token boundary.')
            ids = {}
            for label in string.ascii_uppercase:
                joined = self.tokenizer.encode(suffix + label, add_special_tokens=False)
                if joined[:-1] != suffix_ids:
                    raise ValueError('Each legal letter must be one token after the answer boundary.')
                ids[label] = joined[-1]
            if len(set(ids.values())) != len(ids):
                raise ValueError('Legal answer letters must have distinct token IDs.')
            self.suffix, self.suffix_ids, self.ids = suffix, suffix_ids, ids

        if prefix[-len(self.suffix_ids):] != self.suffix_ids:
            raise ValueError('Actual full-prompt token suffix differs from the cached answer boundary.')
        return [self.ids[label] for label in labels]
