"""Batch independent typed decisions without changing their scoring protocol.

The caller must supply only currently available, independent requests. Future
steps of one closed-loop trajectory cannot be put in the same batch. Timing is
one shared batch measurement, never an isolated per-decision response time.
"""
from dataclasses import dataclass
import json
import time

import numpy as np
import torch


@dataclass(frozen=True)
class ChoiceRequest:
    state: str
    question: dict
    output_keys: list[str]


def _validate(request):
    keys = request.output_keys
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Supply distinct ordered legal choice keys.')
    if request.question.get('type', 'choice') != 'choice':
        raise ValueError('This batch adapter scores action choices, not event probabilities.')
    if set(keys) != set(request.question['criteria']):
        raise ValueError('Question criteria and legal actions differ.')


def _sync(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _normalized(raw, *, rounded=False):
    values = np.asarray(raw, dtype=float)
    tolerance = len(values) * 0.00005 + 1e-8 if rounded else 1e-5
    if (not np.isfinite(values).all() or (values < 0).any()
            or values.sum() <= 0 or abs(values.sum() - 1) > tolerance):
        raise ValueError('Invalid legal-action probability distribution.')
    return (values / values.sum()).tolist()


def _selected_logits(model, tokenizer, encoded, positions, device):
    """Right-pad complete text and score each row at its own causal position.

Keeping the union of requested positions limits the vocabulary projection to
at most batch_size positions, instead of projecting every token in the prompt.
All real tokens retain the positions used in the original serial forward.
"""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Tokenizer needs a pad or EOS token for masked padding.')
    lengths = [int(item['input_ids'].shape[-1]) for item in encoded]
    width = max(lengths)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, (item, length, position) in enumerate(zip(encoded, lengths, positions)):
        if not 0 <= position < length:
            raise ValueError('Scoring position must refer to a real input token.')
        extra = set(item) - {'input_ids', 'attention_mask'}
        if extra:
            raise ValueError(f'Text-only batch received unsupported inputs: {sorted(extra)}')
        input_ids[row, :length] = item['input_ids'][0].to(device)
        attention_mask[row, :length] = item.get('attention_mask', torch.ones_like(item['input_ids']))[0].to(device)
        if not bool(attention_mask[row, position]):
            raise ValueError('Scoring position cannot be padding.')
    keep = torch.tensor(sorted(set(positions)), dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, logits_to_keep=keep).logits
        lookup = {position: index for index, position in enumerate(keep.tolist())}
        selected = torch.stack([logits[row, lookup[position]]
                                for row, position in enumerate(positions)])
    return selected


def _legal_mass(logits, token_ids):
    logits = logits.float()
    return float(torch.exp(torch.logsumexp(logits[token_ids], 0) - torch.logsumexp(logits, 0)))


def _decoder_batch(adapter, requests):
    from examples.qwen35_4b_choice_agent import format_choice_prompt

    prepared = []
    for request in requests:
        if hasattr(adapter, 'format_prompt'):
            prompt, labels = adapter.format_prompt(request.state, request.question, request.output_keys)
            limit = adapter.model.config.max_position_embeddings
        else:
            prompt, labels = format_choice_prompt(adapter.tokenizer, request.state,
                                                   request.question, request.output_keys)
            limit = adapter.max_context_tokens
        inputs = adapter.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        prefix = inputs['input_ids'][0].tolist()
        if len(prefix) > limit:
            raise ValueError('Full supplied history exceeds the model context; no silent truncation.')
        token_ids = []
        for label in labels:
            joined = adapter.tokenizer.encode(prompt + label, add_special_tokens=False)
            if len(joined) != len(prefix) + 1 or joined[:-1] != prefix:
                raise ValueError('Every legal label must be exactly one next token at its answer boundary.')
            token_ids.append(joined[-1])
        if len(set(token_ids)) != len(request.output_keys):
            raise ValueError('Legal outputs must have distinct token ids.')
        prepared.append((inputs, token_ids, prompt, labels))
    logits = _selected_logits(adapter.model, adapter.tokenizer,
                              [row[0] for row in prepared],
                              [int(row[0]['input_ids'].shape[-1]) - 1 for row in prepared],
                              adapter.device)
    results = []
    for request, row, scores in zip(requests, prepared, logits):
        inputs, token_ids, prompt, labels = row
        scores = scores.float()
        allowed = scores[token_ids].double()
        if not bool(torch.isfinite(allowed).all()):
            raise ValueError('Non-finite legal choice logits.')
        raw = torch.softmax(allowed, -1).cpu().tolist()
        probability = _normalized(raw)
        results.append({
            'probabilities': probability, 'raw_probabilities': raw,
            'native_choice': request.output_keys[int(np.argmax(probability))],
            'input_tokens': int(inputs['input_ids'].shape[-1]),
            'allowed_token_mass': _legal_mass(scores, token_ids),
            'allowed_token_mass_semantics': 'Unconditioned vocabulary mass of legal answer symbols at temperature 1.',
            'prompt': prompt, 'label_mapping': dict(zip(labels, request.output_keys)),
        })
    return results


def _intern_batch(adapter, requests):
    from examples.intern_decision_choice_agent import choice_request

    backend = adapter.engine.backend
    official = adapter.official
    prepared = []
    for request in requests:
        row = official.validate_request(choice_request(request.state, request.question, request.output_keys))
        compiled, inputs, positions = backend.encode(row)
        if compiled.fields != ('action',) or len(positions) != 1:
            raise ValueError('Expected exactly the original single action field.')
        ids = [backend.tokenizer.encode(symbol, add_special_tokens=False)
               for symbol in compiled.symbols['action']]
        if any(len(item) != 1 for item in ids):
            raise ValueError('Official candidate symbols must each encode to one token.')
        prepared.append((compiled, inputs, int(positions[0]), [item[0] for item in ids]))
    logits = _selected_logits(backend.model, backend.tokenizer,
                              [row[1] for row in prepared], [row[2] for row in prepared], backend.device)
    results = []
    for request, row, scores in zip(requests, prepared, logits):
        compiled, inputs, position, token_ids = row
        raw_unscaled = torch.softmax(scores[token_ids].float(), -1).cpu().tolist()
        probabilities = dict(zip(request.output_keys, raw_unscaled))
        # Reuse the official temperature and lexical tie-breaking, including its
        # float32 pre-scaling softmax. Do not apply the baseline's float64 readout.
        response = official.scale_result({'answers': {'action': {
            'type': 'choice', 'probabilities': probabilities,
            'choice': official.argmax(probabilities),
        }}}, adapter.engine.temperature)
        answer = response['answers']['action']
        raw = [answer['probabilities'][key] for key in request.output_keys]
        prompt = backend.tokenizer.apply_chat_template(
            compiled.messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True)
        results.append({
            'probabilities': _normalized(raw), 'raw_probabilities': raw,
            'native_choice': answer['choice'], 'input_tokens': int(inputs['input_ids'].shape[-1]),
            'allowed_token_mass': _legal_mass(scores, token_ids),
            'allowed_token_mass_semantics': 'Unconditioned vocabulary mass before official candidate temperature scaling.',
            'prompt': prompt, 'label_mapping': dict(zip(compiled.symbols['action'], request.output_keys)),
            'calibration': response['calibration'], 'scoring_position': position,
        })
    return results


def _laya_batch(adapter, requests):
    agent = adapter.agent
    question = requests[0].question
    if any(request.question != question or list(request.question['criteria']) != list(question['criteria'])
           for request in requests):
        raise ValueError('One native Laya batch requires identical ordered question definitions.')
    internal = agent._to_internal(question)
    limit = int(agent.model.encoder.config.max_position_embeddings)
    head_limit = agent.cfg['head_max_len']
    lengths = []
    for request in requests:
        ids, markers = adapter.build_sequence(agent.tok, request.state, internal,
                                              1000000000, head_limit)
        if len(ids) > limit:
            raise ValueError('Full supplied history exceeds the Laya encoder context; no silent truncation.')
        if len(markers) != len(request.output_keys):
            raise ValueError('Official Laya head budget cannot represent all legal actions.')
        lengths.append(len(ids))
    device_before = agent.device
    responses = agent.predict_batch([request.state for request in requests], {'action': question},
                                     batch_size=len(requests), max_len=limit,
                                     head_max_len=head_limit)
    if agent.device != device_before or agent.device != adapter.requested_device:
        raise RuntimeError('Laya changed device during batch inference; timing is invalid.')
    if len(responses) != len(requests):
        raise ValueError('Laya returned an incomplete batch.')
    results = []
    for request, response, length in zip(requests, responses, lengths):
        if int(response['usage']['input_tokens']) != length:
            raise ValueError('Runtime Laya input length differs from the full-input preflight.')
        answer = response['answers']['action']
        raw = [answer['probabilities'][key] for key in request.output_keys]
        if answer['choice'] not in request.output_keys:
            raise ValueError('Laya returned a non-legal native choice.')
        results.append({
            'probabilities': _normalized(raw, rounded=True), 'raw_probabilities': raw,
            'native_choice': answer['choice'], 'input_tokens': length,
            'allowed_token_mass': None,
            'allowed_token_mass_semantics': 'Not applicable: official Laya choice head has no full-vocabulary distribution.',
        })
    return results


def choice_probabilities_batch(adapter, requests, batch_size=8):
    """Return one metadata/probability dict per ChoiceRequest in original order.

    Weights, precision, prompts and calibration belong to the supplied adapter;
    this function never changes them. Laya groups identical ordered schemas;
    decoder batches may contain different schemas and different input lengths.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    requests = list(requests)
    if not requests:
        return []
    for request in requests:
        _validate(request)
    if hasattr(adapter, 'agent') and hasattr(adapter, 'build_sequence'):
        run = _laya_batch
        device = adapter.agent.device
        groups = {}
        for index, request in enumerate(requests):
            # Do not sort schema keys: changing option order changes the prompt.
            key = json.dumps(request.question, ensure_ascii=False, sort_keys=False)
            groups.setdefault(key, []).append(index)
        groups = list(groups.values())
    elif hasattr(adapter, 'engine') and hasattr(adapter, 'official'):
        run = _intern_batch
        device = adapter.engine.backend.device
        groups = [list(range(len(requests)))]
    elif hasattr(adapter, 'format_prompt') or hasattr(adapter, 'max_context_tokens'):
        run = _decoder_batch
        device = adapter.device
        groups = [list(range(len(requests)))]
    else:
        raise TypeError('Unsupported typed-choice adapter; no implicit serial fallback.')
    results = [None] * len(requests)
    batch_id = 0
    for indices in groups:
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            _sync(device)
            began = time.perf_counter()
            outputs = run(adapter, [requests[index] for index in selected])
            _sync(device)
            elapsed = time.perf_counter() - began
            for index, output in zip(selected, outputs):
                output.update(request_index=index, batch_id=batch_id, batch_size=len(selected),
                              batch_elapsed_seconds=elapsed,
                              batch_amortized_seconds=elapsed / len(selected),
                              timing_scope='Shared batch tokenization, full-input checks, forward and readout; excludes model load. Not isolated request latency.')
                results[index] = output
            batch_id += 1
    adapter.max_input_tokens = max(adapter.max_input_tokens, max(row['input_tokens'] for row in results))
    return results
