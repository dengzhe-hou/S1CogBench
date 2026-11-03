"""Explicit event-probability adapters; action distributions are not posteriors."""
import importlib.metadata
import math
import re


class LayaEventProbability:
    """Keep the deployed English SDK's default context and noul calibration."""

    def __init__(self, device='cpu'):
        import torch
        from src.choice_models import LayaChoice

        torch.set_num_threads(4)
        self.base = LayaChoice(device=device)
        self.base.agent.dtype = torch.float32
        self.max_input_tokens = 0
        self.metadata = {
            'model': 'local_laya_english', 'primitive': 'noul',
            'precision': 'FP32 weights and forward; matches earlier Laya task runs.',
            'actual_parameter_dtypes': sorted({str(p.dtype) for p in self.base.agent.model.parameters()}),
            'sdk_config': self.base.agent.cfg,
            'temperature_by_options': self.base.agent.temperature_by_options,
            'training': 'Frozen checkpoint; no task fitting, calibration or prompt selection.',
            'event_semantics': 'SDK noul is P(true) for the stated origin event, not action.act_probability.',
            'probability_precision': 'Preserve native SDK four-decimal noul; two-decimal report is a separate view.',
            'labels': 'Default model-facing false/true; no label overrides.',
            'context': 'Default max_len=512 and head_max_len=192; fail on any truncation.',
            'interface_adaptation': 'Original numerical posterior task rendered as an explicit event question.',
            'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers']},
        }

    def posterior_probability(self, state, question):
        if question['type'] != 'noul' or 'labels' in question:
            raise ValueError('Use the explicit event question with default noul labels.')
        agent = self.base.agent
        internal = agent._to_internal(question)
        ids, markers = self.base.build_sequence(agent.tok, state, internal,
                                               agent.cfg['max_len'], agent.cfg['head_max_len'])
        full_ids, full_markers = self.base.build_sequence(agent.tok, state, internal, 8192, 8192)
        if ids != full_ids or markers != full_markers or len(markers) != 2:
            raise ValueError('Event question or visible information truncated; do not shorten inputs.')
        self.last_input_tokens = len(ids)
        self.max_input_tokens = max(self.max_input_tokens, len(ids))
        answer = agent.predict(state, {'event': question})['answers']['event']
        if agent.device != self.base.requested_device:
            raise RuntimeError('Laya changed device; do not substitute CPU timing for GPU timing.')
        probability = float(answer['noul'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('The SDK must return an event probability in [0,1].')
        self.last_sdk_answer = answer
        self.last_raw_answer = answer
        return probability


class BayesEventProbability:
    """Compute the reference from the same visible V1 counts supplied to models."""

    metadata = {'model': 'exact_Bayes_visible_information_reference',
                'inputs': 'Original visible wheel/urn counts, observed color and named target event.',
                'training': 'No fitted parameters; no hidden urn or stored posterior used.',
                'probability_precision': 'Unrounded analytical posterior.'}

    def posterior_probability(self, state, question):
        target = re.search(r'urn ([FJ])', question['instructions']).group(1)
        wheel = re.search(
            r'wheel of fortune contains (\d+) sections? labelled ([FJ]) and '
            r'(\d+) sections? labelled ([FJ])', state, re.IGNORECASE)
        if wheel is None:
            raise ValueError('Expected original V1 wheel counts in the visible prompt.')
        sections = {wheel.group(2): int(wheel.group(1)), wheel.group(4): int(wheel.group(3))}
        urns = {name: (int(red), int(blue)) for name, red, blue in re.findall(
            r'urn ([FJ]) contains \((\d+), (\d+)\)', state, re.IGNORECASE)}
        color = re.search(r'A (red|blue)\s+ball was drawn\.', state).group(1)
        prior = sections[target] / sum(sections.values())
        other = next(name for name in sections if name != target)
        target_red = urns[target][0] / sum(urns[target])
        other_red = urns[other][0] / sum(urns[other])
        if abs(target_red + other_red - 1) > 1e-14:
            raise ValueError('Original V1 has complementary urn color probabilities.')
        # Same arithmetic as the original scorer; use the visible red counts
        # and complement them for blue, avoiding a rounding-only disagreement.
        target_likelihood = target_red if color == 'red' else 1 - target_red
        other_likelihood = 1 - target_likelihood
        return prior * target_likelihood / (prior * target_likelihood + (1 - prior) * other_likelihood)
