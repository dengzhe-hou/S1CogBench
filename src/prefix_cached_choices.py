"""Token-exact prefix reuse for the existing standard-attention Llama adapters.

Only the shared causal prefix is retained. The question/chat/answer tail from a
previous decision is removed whenever it is absent from the new complete prompt.
This preserves the intended inputs, not a claim of bitwise numerical equality.
"""
import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from src.hf_choice_models import FrozenDecoderChoice


class PrefixCachedDecoderChoice:
    """Wrap an already loaded Llama/Minitaur FrozenDecoderChoice; no reload."""

    def __init__(self, adapter):
        if (not isinstance(adapter, FrozenDecoderChoice)
                or not isinstance(adapter.model, LlamaForCausalLM)
                or adapter.model.config.model_type != 'llama'):
            raise TypeError('Prefix caching is limited to the current standard-attention Llama/Minitaur adapters.')
        if adapter.model.training:
            raise ValueError('Use the original frozen eval-mode model.')
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = adapter.tokenizer
        self.device = adapter.device
        self.max_input_tokens = 0
        self.metadata = dict(adapter.metadata)
        self.metadata['prefix_cache'] = (
            'Exact token LCP, standard Llama DynamicCache crop; original full prompt and legal readout retained. '
            'Numerical agreement and speed require paired measurement.')
        self.reset()

    def reset(self):
        """Drop all retained history, e.g. when changing participants."""
        self.cache = DynamicCache(config=self.model.config)
        if any(type(layer) is not DynamicLayer for layer in self.cache.layers):
            raise TypeError('Only standard full-attention DynamicCache layers are supported.')
        self.cached_token_ids = ()
        self.last_lcp_tokens = 0
        self.last_reused_tokens = 0
        self.last_fresh_tokens = 0
        self.last_cache_tokens = 0
        self.last_cache_bytes = 0
        self.last_cache_info = {
            'lcp_tokens': 0, 'reused_tokens': 0, 'fresh_tokens': 0,
            'cache_tokens': 0, 'cache_bytes': 0,
            'memory_scope': 'Logical K/V tensor payload only; not CUDA peak or allocator-reserved memory.',
        }

    def format_prompt(self, state, question, output_keys):
        return self.adapter.format_prompt(state, question, output_keys)

    def choice_probabilities(self, state, question, output_keys):
        if not output_keys or len(set(output_keys)) != len(output_keys):
            raise ValueError('Supply distinct ordered legal choices.')
        if set(output_keys) != set(question['criteria']):
            raise ValueError('Question criteria and legal choices differ.')
        prompt, labels = self.format_prompt(state, question, output_keys)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        prefix = inputs['input_ids'][0].tolist()
        token_ids = []
        for label in labels:
            joined = self.tokenizer.encode(prompt + label, add_special_tokens=False)
            if len(joined) != len(prefix) + 1 or joined[:-1] != prefix:
                raise ValueError('Each legal output must be one next token after the actual prompt.')
            token_ids.append(joined[-1])
        if len(set(token_ids)) != len(output_keys):
            raise ValueError('Legal outputs must have distinct token ids.')
        length = len(prefix)
        if length > self.model.config.max_position_embeddings:
            raise ValueError('Full history exceeds the model context; no silent truncation.')
        if not length:
            raise ValueError('The original prompt must contain at least one token.')
        if set(inputs) - {'input_ids', 'attention_mask'}:
            raise ValueError('Expected the original Llama text tokenizer fields only.')
        mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        if not bool(mask.all()):
            raise ValueError('A single complete original prompt must not contain padding.')

        common = 0
        for old, new in zip(self.cached_token_ids, prefix):
            if old != new:
                break
            common += 1
        # Even an identical or shorter-prefix request recomputes its last token
        # so logits are obtained at the exact original answer boundary.
        keep = min(common, length - 1)
        if self.cache.get_seq_length() != len(self.cached_token_ids):
            raise RuntimeError('Cache length differs from its recorded token prefix; reset before reuse.')
        self.cache.crop(keep)
        self.cached_token_ids = tuple(prefix[:keep])
        fresh_ids = inputs['input_ids'][:, keep:].to(self.device)
        positions = torch.arange(keep, length, device=self.device)
        try:
            with torch.inference_mode():
                result = self.model(
                    input_ids=fresh_ids, attention_mask=mask.to(self.device),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                self.cache = result.past_key_values
                logits = result.logits[0, -1].float()
        except RuntimeError:
            # A failed forward may have updated only some layers. Keep the
            # original failure visible and never reuse that partial cache.
            self.reset()
            raise
        if self.cache.get_seq_length() != length:
            raise RuntimeError('Llama returned an incomplete cache for the full supplied prompt.')
        self.cached_token_ids = tuple(prefix)

        selected = logits[token_ids].double()
        if not bool(torch.isfinite(selected).all()):
            raise ValueError('Non-finite legal choice logits.')
        probability = torch.softmax(selected, -1).cpu().numpy()
        probability /= probability.sum()
        self.last_input_tokens = length
        self.max_input_tokens = max(self.max_input_tokens, length)
        self.last_prompt = prompt
        self.last_label_mapping = dict(zip(labels, output_keys))
        self.last_allowed_token_mass = float(torch.exp(
            torch.logsumexp(logits[token_ids], 0) - torch.logsumexp(logits, 0)))
        self.last_native_choice = output_keys[int(np.argmax(probability))]
        self.last_lcp_tokens = common
        self.last_reused_tokens = keep
        self.last_fresh_tokens = length - keep
        self.last_cache_tokens = length
        self.last_cache_bytes = sum(tensor.numel() * tensor.element_size()
                                    for layer in self.cache.layers
                                    for tensor in (layer.keys, layer.values)
                                    if tensor is not None)
        self.last_cache_info = {
            'lcp_tokens': common, 'reused_tokens': keep,
            'fresh_tokens': length - keep, 'cache_tokens': length,
            'cache_bytes': self.last_cache_bytes,
            'memory_scope': 'Logical K/V tensor payload only; not CUDA peak or allocator-reserved memory.',
        }
        return probability
