"""Template: a Hugging Face causal language model as an S1CogBench adapter.

Copy this file, then run, for example:

    S1_HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \\
    python -m s1cogbench eval --adapter my_hf_agent.py --device cuda:0 \\
        --model-label my-llama --tasks all --output runs/my-llama

S1_HF_MODEL is a local directory or Hub id; S1_HF_DTYPE (default bfloat16)
and S1_HF_REVISION are optional. Report the revision, dtype and this readout
with your results (docs/SUBMIT_RESULTS.md).

Readout (the one used for the accepted Llama, Minitaur and Qwen choice rows,
src/hf_choice_models.py and src/integer_label_choice.py):
- Prompt: the visible state, the question's instructions, then either one line
  "label: key - criterion" per legal option and "Choose one of the listed
  option letters.", or for integer keys "Legal integer reports: ..." and
  "Return only one legal integer.". Labels are the task keys when they are
  single capital letters or integers, otherwise A, B, C, ... in legal order.
  Chat models get their chat template (with thinking disabled where the
  template supports it); base models get the raw text and "Answer: ".
- When every label is one token after the prompt, choice_probabilities is the
  softmax at temperature 1 over the legal label logits of the next token.
  Otherwise each legal reply (label plus end of turn) is scored by the sum of
  its token log-probabilities, without length normalization, and the softmax
  is taken over those sums.
- The native choice is the argmax. Nothing is generated and the state is never
  truncated: a prompt longer than the model context raises an error.
- posterior_probability (Bayesian urn, nonliteral numbers) greedily generates
  at most 6 tokens after "Return only the probability as a number from 0 to
  1." and accepts only a reply that is such a number; anything else is None
  (invalid). confidence_probability (Restless bandit) asks the same for
  "Machine <chosen> currently has the higher average reward.".

This is an explicit typed-choice readout, not the model's free generation and
not the original full-vocabulary NLL of any source paper.
"""
import copy
import os
import re
import string

PROBABILITY_READOUT = True
MAX_NEW_NUMERIC_TOKENS = 6
PROBABILITY = re.compile(r'\s*(0(?:\.\d+)?|1(?:\.0*)?|\.\d+)\s*')


class HFCausalLMChoice:
    def __init__(self, model, device, dtype='bfloat16', revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        self.chat = self.tokenizer.chat_template is not None
        self.context = getattr(self.model.config, 'max_position_embeddings', None)
        self.turn_end = self._turn_end()
        self.metadata = {
            'model': model, 'revision': revision, 'dtype': dtype, 'device': str(self.device),
            'prompt_format': 'chat_template' if self.chat else 'raw_text',
            'turn_end': self.turn_end,
            'readout': 'Temperature-one softmax over legal replies; one-token labels use next-token logits, '
                       'multi-token replies the summed reply log-probability with end of turn.',
            'native_choice': 'argmax of the legal probabilities',
            'training': 'Frozen; no task fitting.',
        }

    # ------------------------------------------------------------ prompts
    def _render(self, content):
        if not self.chat:
            return content + '\nAnswer: '
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    def _turn_end(self):
        if not self.chat:
            return self.tokenizer.eos_token or ''
        probe = 'S1COGBENCH_REPLY_PROBE'
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'u'}, {'role': 'assistant', 'content': probe}],
            tokenize=False, enable_thinking=False)
        return text.split(probe, 1)[1].strip() or (self.tokenizer.eos_token or '')

    @staticmethod
    def labels(keys):
        keys = [str(k) for k in keys]
        if all(len(k) == 1 and k in string.ascii_uppercase for k in keys) or all(k.isdigit() for k in keys):
            return keys
        if len(keys) > 26:
            raise ValueError('More than 26 non-integer legal options; extend the label scheme.')
        return list(string.ascii_uppercase[:len(keys)])

    def content(self, state, question, keys, labels):
        if not isinstance(state, str):
            raise ValueError('This template expects a text state.')
        if all(str(k).isdigit() for k in keys):
            return (state + '\n' + question['instructions'] + '\nLegal integer reports: '
                    + ', '.join(labels) + '.\nReturn only one legal integer.')
        options = '\n'.join(f'{label}: {key} - {question["criteria"][key]}' for label, key in zip(labels, keys))
        return (state + '\n' + question['instructions'] + '\n' + options
                + '\nChoose one of the listed option letters.')

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=not self.chat)['input_ids']

    # ------------------------------------------------------------ choice
    def choice_probabilities(self, state, question, output_keys):
        torch = self.torch
        keys = list(output_keys)
        labels = self.labels(keys)
        prompt = self._render(self.content(state, question, keys, labels))
        prefix = self._ids(prompt)
        replies = []
        for label in labels:
            joined = self._ids(prompt + label)
            if joined[:len(prefix)] != prefix:
                raise ValueError('A legal label changes the tokenization of the prompt.')
            replies.append(joined[len(prefix):])
        single = all(len(r) == 1 for r in replies) and len({r[0] for r in replies}) == len(replies)
        if not single:
            replies = []
            for label in labels:
                joined = self._ids(prompt + label + self.turn_end)
                if joined[:len(prefix)] != prefix:
                    raise ValueError('A legal reply changes the tokenization of the prompt.')
                replies.append(joined[len(prefix):])
            if len({tuple(r) for r in replies}) != len(replies):
                raise ValueError('Legal replies must be distinct token sequences.')
        longest = max(map(len, replies))
        if self.context and len(prefix) + longest > self.context:
            raise ValueError(f'Prompt of {len(prefix)} tokens exceeds the model context; no truncation.')
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([prefix], device=self.device), use_cache=not single)
            first = torch.log_softmax(out.logits[0, -1].double(), -1)
            if single:
                scores = first[[r[0] for r in replies]]
            else:
                scores = []
                for reply in replies:
                    total = first[reply[0]]
                    if len(reply) > 1:
                        branch = self.model(input_ids=torch.tensor([reply[:-1]], device=self.device),
                                            past_key_values=copy.deepcopy(out.past_key_values),
                                            use_cache=True)
                        logp = torch.log_softmax(branch.logits[0].double(), -1)
                        total = total + sum(logp[i, reply[i + 1]] for i in range(len(reply) - 1))
                    scores.append(total)
                scores = torch.stack(scores)
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite legal reply score.')
            p = torch.softmax(scores, -1).cpu().tolist()
        self.last_input_tokens = len(prefix)
        self.last_label_mapping = dict(zip(labels, keys))
        self.last_native_choice = keys[max(range(len(keys)), key=p.__getitem__)]
        return p

    # ------------------------------------------------------------ numeric reports
    def _probability(self, content):
        torch = self.torch
        prompt = self._render(content + '\nReturn only the probability as a number from 0 to 1. '
                                        'Do not add an explanation.')
        prefix = self._ids(prompt)
        if self.context and len(prefix) + MAX_NEW_NUMERIC_TOKENS > self.context:
            raise ValueError('Prompt exceeds the model context; no truncation.')
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=torch.tensor([prefix], device=self.device),
                attention_mask=torch.ones(1, len(prefix), dtype=torch.long, device=self.device),
                max_new_tokens=MAX_NEW_NUMERIC_TOKENS, do_sample=False,
                pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None
                              else self.tokenizer.pad_token_id))
        text = self.tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=True)
        self.last_input_tokens = len(prefix)
        self.last_raw_answer = text
        match = PROBABILITY.fullmatch(text)
        return float(match.group(1)) if match else None

    def posterior_probability(self, state, question):
        return self._probability(state + '\n' + question['instructions'])

    def confidence_probability(self, state, chosen, original_specifications):
        return self._probability(state + f'\nMachine {chosen} currently has the higher average reward.')


def create_agent(device):
    model = os.environ.get('S1_HF_MODEL')
    if not model:
        raise SystemExit('Set S1_HF_MODEL to a local model directory or Hugging Face Hub id.')
    return HFCausalLMChoice(model, device, os.environ.get('S1_HF_DTYPE', 'bfloat16'),
                            os.environ.get('S1_HF_REVISION'))
