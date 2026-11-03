#!/usr/bin/env python3
"""Faithful memory-efficient Centauri scoring; original dense entry is retained.

Keep full causal context, masks, BF16 base, default PEFT cast, original group
aggregation and 32768 cap. Compute vocabulary logits only at shifted response
positions; chunk CE. This is numeric equivalence, not a bitwise claim.
"""
import argparse
from collections import Counter
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_psych_official_local import (
    ADAPTER, BASE, MAX_LENGTH, SOURCES, check_assets, dtype_record,
    load_local_data, load_official, save, tokenizer_for,
)


def active_positions(labels):
    if labels.shape[0] != 1:
        raise ValueError('Original evaluator uses batch size one.')
    shifted = labels[0, 1:]
    positions = (shifted != -100).nonzero(as_tuple=True)[0]
    if not positions.numel():
        raise ValueError('A real sequence must contain supervised response tokens.')
    return positions, shifted[positions]


def token_losses(logits, targets, chunk_size, float32):
    import torch
    import torch.nn.functional as F
    losses = []
    for begin in range(0, len(targets), chunk_size):
        chunk = logits[begin:begin + chunk_size]
        if float32:
            chunk = chunk.float()
        losses.append(F.cross_entropy(chunk, targets[begin:begin + chunk_size].to(chunk.device),
                                      reduction='none'))
    return torch.cat(losses)


def response_sums(logits, labels, chunk_size):
    """Original CPU CE dtype and sequential span accumulation, without prompts."""
    import torch
    positions, targets = active_positions(labels.cpu())
    losses = token_losses(logits[0].cpu(), targets, chunk_size, float32=False)
    result = []
    item_loss = 0
    last = None
    for position, loss in zip(positions.tolist(), losses):
        if last is not None and position != last + 1:
            result.append(item_loss)
            item_loss = 0
        item_loss += loss
        last = position
    result.append(item_loss)
    # Original preprocess returns torch.Tensor(total_loss), hence FP32 output
    # even when the within-span CPU CE/additions used BF16.
    return torch.tensor([float(value) for value in result], dtype=torch.float32)


def sparse_trainer_class(original_class, chunk_size):
    class ResponseOnlyTrainer(original_class):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            from transformers.modeling_outputs import CausalLMOutputWithPast
            labels = inputs['labels']
            positions, targets = active_positions(labels)
            # CPU tensor indices are valid for the last-device hidden states in
            # Qwen3's tensor-index logits_to_keep, including multi-GPU placement.
            model_inputs = {key: value for key, value in inputs.items()
                            if key in ['input_ids', 'attention_mask', 'position_ids']}
            outputs = model(**model_inputs, use_cache=False, logits_to_keep=positions.cpu())
            logits = outputs.logits
            if logits.shape[1] != positions.numel():
                raise ValueError('Qwen3 must honor tensor response-position logits_to_keep.')
            ce = token_losses(logits[0], targets, chunk_size, float32=True)
            loss = ce.mean()
            correct = (logits[0].argmax(-1) == targets.to(logits.device)).sum()
            accuracy = (correct.float() / targets.numel()).item()
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['mean_token_accuracy'].append(accuracy)
            # Full-prompt entropy is a Trainer auxiliary diagnostic, not an
            # original task CSV metric. Do not invent it from response logits.
            selected = CausalLMOutputWithPast(loss=loss, logits=logits)
            return (loss, selected) if return_outputs else loss
    return ResponseOnlyTrainer


def memory_estimate():
    from safetensors import safe_open
    assets = check_assets()
    adapter_elements = 0
    with safe_open(ADAPTER / 'adapter_model.safetensors', framework='pt', device='cpu') as weights:
        for name in weights.keys():
            adapter_elements += math.prod(weights.get_slice(name).get_shape())
    return {'base_bf16_weight_bytes': assets['base_weight_bytes'],
            'adapter_default_fp32_upper_bytes': adapter_elements * 4,
            'one_ce_chunk64_fp32_bytes': 64 * 151936 * 4,
            'response_logits_bytes_per_response_token_bf16': 151936 * 2,
            'response_logits_bytes_per_response_token_fp32': 151936 * 4,
            'context_hidden_state_bf16_bytes': MAX_LENGTH * 5120 * 2,
            'activation_estimate': 'About 4 GiB per participating GPU reserved; actual longest-context '
                'peak is measured by a separate GPU sanity. No dense prompt vocabulary logits.',
            'kind': 'Static tensor/weight estimate, not measured GPU memory.'}


def gpu_budget(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA run requires available GPUs; no CPU-offload fallback.')
    explicit = json.loads(args.gpu_max_memory) if args.gpu_max_memory else {}
    estimate = memory_estimate()
    # PEFT attaches adapters after the base-only automatic placement. Reserve
    # the entire actual FP32 adapter upper bound on each participating card so
    # it cannot consume the separately declared activation margin.
    adapter_reserve = estimate['adapter_default_fp32_upper_bytes'] if args.model == 'qwentaur14b-r64' else 0
    limits, observed = {}, []
    reserve = int(args.activation_reserve_gib * 2**30)
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        available = free - reserve - adapter_reserve
        if str(index) in explicit:
            cap = int(float(explicit[str(index)]) * 2**30)
            if cap > available:
                raise ValueError('Explicit GPU weight cap exceeds live free memory minus activation reserve.')
            available = cap
        if available <= 0:
            raise RuntimeError('Insufficient live GPU memory after activation reserve; keep queued.')
        limits[index] = available
        observed.append({'logical_gpu': index, 'free_bytes': free, 'total_bytes': total,
                         'weight_cap_bytes': available, 'activation_reserve_bytes': reserve,
                         'adapter_reserve_bytes': adapter_reserve})
    if sum(limits.values()) < estimate['base_bf16_weight_bytes']:
        raise RuntimeError('Insufficient total GPU memory for original weights; keep queued.')
    return limits, observed, estimate


def load_gpu_model(args, output):
    import torch
    from accelerate import init_empty_weights, infer_auto_device_map
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import PeftModel
    limits, observed, estimate = gpu_budget(args)
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    device_map = infer_auto_device_map(empty, max_memory=limits, dtype=torch.bfloat16,
                                     no_split_module_classes=['Qwen3DecoderLayer'])
    del empty
    if any(value in ['cpu', 'disk'] for value in device_map.values()):
        raise RuntimeError('Original model does not fit GPUs without CPU/disk offload; keep queued.')
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True,
                        dtype=torch.bfloat16, device_map=device_map, max_memory=limits)
    if args.model == 'qwentaur14b-r64':
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    underlying = model.get_base_model() if args.model == 'qwentaur14b-r64' else model
    if 'logits_to_keep' not in inspect.signature(underlying.forward).parameters:
        raise TypeError('Installed Qwen3 must support tensor logits_to_keep; no compatibility fallback.')
    actual = {**dtype_record(model, args.model), 'hf_device_map': device_map,
              'live_gpu_memory': observed, 'static_memory_estimate': estimate,
              'attention_implementation': underlying.config._attn_implementation,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if actual['quantized']:
        raise ValueError('Keep original nonquantized weights.')
    save(output / 'actual_model.json', actual)
    return model, actual


def install_sparse(module, args):
    module.SFTTrainer = sparse_trainer_class(module.SFTTrainer, args.ce_chunk_size)
    module.preprocess_logits_for_metrics = lambda logits, labels: response_sums(logits, labels, args.ce_chunk_size)


def cpu_witness(output):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    module = load_official('psych101')
    tokenizer = tokenizer_for('qwentaur14b-r64')
    left, right = module.get_response_template_ids(tokenizer)
    collator = module.DataCollatorForCompletionOnlyLM(response_template=left,
                    instruction_template=right, tokenizer=tokenizer)
    texts = ['>> Pick A or B. <<A>> Pick again. <<B>>',
             '>> Report several words. << one two three>> Report a number. <<48.0>>',
             '>> One more response. <<A>>']
    torch.manual_seed(20261009)
    config = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=MAX_LENGTH)
    records = []
    for dtype in [torch.float32, torch.bfloat16]:
        model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
        seq_dense, seq_sparse, accuracies_dense, accuracies_sparse = [], [], [], []
        responses_dense, responses_sparse = [], []
        for index, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors=None)
            batch = collator([encoded])
            labels = batch['labels']
            positions, targets = active_positions(labels)
            with torch.inference_mode():
                dense = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                              labels=labels, use_cache=False)
                sparse = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               logits_to_keep=positions, use_cache=False)
            selected_dense = dense.logits[:, positions]
            torch.testing.assert_close(selected_dense, sparse.logits, rtol=1e-5, atol=1e-5)
            sparse_mean = token_losses(sparse.logits[0], targets, 2, float32=True).mean()
            torch.testing.assert_close(dense.loss, sparse_mean, rtol=1e-5, atol=1e-5)
            original_sums = module.preprocess_logits_for_metrics(dense.logits, labels)
            optimized_sums = response_sums(sparse.logits, labels, 2)
            if index < 2 and len(original_sums) < 2:
                raise ValueError('Witness must cover multiple separated response spans.')
            torch.testing.assert_close(original_sums, optimized_sums, rtol=1e-5, atol=1e-5)
            # Force some synthetic witness targets correct so a bogus unshifted
            # accuracy cannot pass merely because random-model accuracy is zero.
            witness_labels = labels.clone()
            witness_labels[0, positions[::2] + 1] = selected_dense[0, ::2].argmax(-1)
            _, witness_targets = active_positions(witness_labels)
            correct_dense = (dense.logits[0, :-1].argmax(-1)[positions] == witness_targets).float().mean()
            correct_sparse = (sparse.logits[0].argmax(-1) == witness_targets).float().mean()
            torch.testing.assert_close(correct_dense, correct_sparse, rtol=0, atol=0)
            seq_dense.append(dense.loss.float())
            seq_sparse.append(sparse_mean.float())
            accuracies_dense.append(correct_dense)
            accuracies_sparse.append(correct_sparse)
            responses_dense.extend(original_sums.tolist())
            responses_sparse.extend(optimized_sums.tolist())
            records.append({'dtype': str(dtype), 'sequence': index, 'context_tokens': len(encoded['input_ids']),
                            'response_tokens': len(targets), 'response_spans': len(original_sums),
                            'dense_mean_ce': dense.loss.item(), 'selected_mean_ce': sparse_mean.item(),
                            'mean_ce_absolute_difference': abs(dense.loss.item() - sparse_mean.item()),
                            'correct_masked_token_accuracy': correct_dense.item(),
                            'max_response_sum_absolute_difference': float((original_sums - optimized_sums).abs().max())})
        torch.testing.assert_close(torch.stack(seq_dense).mean(), torch.stack(seq_sparse).mean(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.tensor(responses_dense).mean(), torch.tensor(responses_sparse).mean(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.stack(accuracies_dense).mean(), torch.stack(accuracies_sparse).mean(), rtol=0, atol=0)
    pipeline = trainer_pipeline_witness(module, tokenizer, config, texts, output)
    result = {'status': 'cpu_dense_selected_equivalence_passed', 'witness_cases': records,
              'original_trainer_pipeline_witness': pipeline,
              'original_group_aggregation': 'Standard loss/accuracy: equal sequence mean. '
                  'Variable-length: per-response token sums, equal mean over all responses.',
              'numeric_tolerance': 'rtol=1e-5, atol=1e-5; not bitwise equivalence.',
              'synthetic_dependency_witness_only': True, 'loaded_14b_weights': False,
              'full_evaluation_complete': False, 'memory_estimate': memory_estimate()}
    save(output / 'summary.json', result)
    save(output / 'status.json', result)
    print(json.dumps(result), flush=True)


def trainer_pipeline_witness(module, tokenizer, model_config, texts, output):
    """Exercise inherited original add-EOS/tokenization/mask and group metrics."""
    import torch
    from datasets import Dataset
    from transformers import Qwen3ForCausalLM
    original_class = module.SFTTrainer
    raw_rows = []
    for task in SOURCES:
        dataset = load_local_data(task, output)['test']
        # Short real rows are an implementation witness, not a reduced task.
        candidate = min(dataset, key=lambda row: len(row['text']))
        raw_rows.append(candidate['text'])
    fixture = Dataset.from_dict({'text': texts + raw_rows})
    left, right = module.get_response_template_ids(tokenizer)
    collator = module.DataCollatorForCompletionOnlyLM(response_template=left,
                    instruction_template=right, tokenizer=tokenizer)
    torch.manual_seed(20261009)
    model = Qwen3ForCausalLM(model_config).eval()
    results = []
    for variable in [False, True]:
        trainers = []
        for sparse in [False, True]:
            args = module.SFTConfig(output_dir=str(output / ('pipeline_' + str(variable) + '_' + str(sparse))),
                    per_device_eval_batch_size=1, report_to='none', dataset_text_field='text',
                    max_length=MAX_LENGTH, dataset_num_proc=1, use_cpu=True, bf16=False, fp16=False,
                    eval_accumulation_steps=1)
            trainer_class = sparse_trainer_class(original_class, 2) if sparse else original_class
            trainer = trainer_class(model=model, processing_class=tokenizer, args=args,
                         train_dataset=fixture, eval_dataset=fixture, data_collator=collator,
                         compute_metrics=module.compute_metrics_per_response if variable else None,
                         preprocess_logits_for_metrics=(
                             (lambda logits, labels: response_sums(logits, labels, 2)) if sparse
                             else module.preprocess_logits_for_metrics) if variable else None)
            trainers.append(trainer)
        for original, selected in zip(trainers[0].eval_dataset, trainers[1].eval_dataset):
            if original['input_ids'] != selected['input_ids']:
                raise ValueError('Original and sparse inherited SFT preprocessing must match including EOS.')
            a = collator([{'input_ids': original['input_ids']}])
            b = collator([{'input_ids': selected['input_ids']}])
            torch.testing.assert_close(a['labels'], b['labels'], rtol=0, atol=0)
        dense_metrics = trainers[0].evaluate()
        sparse_metrics = trainers[1].evaluate()
        key = 'eval_custom_loss' if variable else 'eval_loss'
        for metric in ['eval_loss', key, 'eval_mean_token_accuracy']:
            torch.testing.assert_close(torch.tensor(dense_metrics[metric]),
                                       torch.tensor(sparse_metrics[metric]), rtol=1e-5, atol=1e-5)
        results.append({'variable_length': variable, 'synthetic_rows': len(texts), 'real_source_rows': len(raw_rows),
                        'original_eos_and_tokens_equal': True, 'original_collator_labels_equal': True,
                        'dense_group_loss': dense_metrics[key], 'selected_group_loss': sparse_metrics[key],
                        'group_loss_absolute_difference': abs(dense_metrics[key] - sparse_metrics[key]),
                        'dense_equal_sequence_accuracy': dense_metrics['eval_mean_token_accuracy'],
                        'selected_equal_sequence_accuracy': sparse_metrics['eval_mean_token_accuracy']})
    return results


def evaluate(args, output):
    import pandas as pd
    prepared = json.loads((args.prepared_dir / 'config.json').read_text())
    audit = json.loads((args.prepared_dir / 'data_mask_audit.json').read_text())
    if (prepared['status'] != 'prepared_cpu' or prepared['task'] != args.task
            or prepared['model'] != args.model or audit['rows'] != SOURCES[args.task]['rows']):
        raise ValueError('Use the completed same-model, same-task original CPU preparation.')
    module = load_official(args.task)
    dataset = load_local_data(args.task, output)
    model, actual = load_gpu_model(args, output)
    install_sparse(module, args)
    module.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *a, **k: model)
    module.load_dataset = lambda *a, **k: dataset
    configuration = {'task': args.task, 'model': args.model, 'implementation': 'faithful_memory_efficient_scorer',
              'prepared_directory': str(args.prepared_dir), 'max_context': MAX_LENGTH, 'use_cache': False,
              'ce_chunk_size': args.ce_chunk_size, 'actual_model': actual,
              'source_rows_exceeding_original_cap': audit['source_rows_exceeding_original_cap'],
              'numeric_equivalence': 'Same loss target and group aggregation; floating reduction order may differ.',
              'auxiliary_prompt_entropy': 'Not computed; not an original task CSV metric.',
              'protocol': 'Original all rows/groups/tokenizer/collator/SFT preprocessing/evaluate functions.'}
    save(output / 'config.json', configuration)
    save(output / 'status.json', {**configuration, 'status': 'running'})
    start = time.monotonic()
    if args.gpu_sanity:
        return gpu_sanity(args, output, module, dataset, model, configuration)
    module.evaluate(model_path=str(ADAPTER if args.model == 'qwentaur14b-r64' else BASE),
                    output_dir=str(output / 'results'), load_in_4bit=False,
                    device_map='auto', max_seq_length=MAX_LENGTH, backend='transformers')
    csv_path = output / 'results' / ((ADAPTER if args.model == 'qwentaur14b-r64' else BASE).name + '.csv')
    frame = pd.read_csv(csv_path)
    if len(frame) != SOURCES[args.task]['groups'] or set(frame.task) != set(module.TASK_NAMES):
        raise ValueError('Every original task group must finish before reporting complete.')
    summary = {**configuration, 'status': 'complete', 'original_rows': len(dataset['test']),
               'task_groups': len(frame), 'results_csv': str(csv_path),
               'elapsed_seconds': time.monotonic() - start}
    save(output / 'summary.json', summary)
    save(output / 'status.json', summary)
    print(json.dumps(summary), flush=True)


def gpu_sanity(args, output, module, dataset, model, configuration):
    """A separate real 32768-context witness, never a full paper result."""
    import torch
    from datasets import Dataset
    tokenizer = tokenizer_for(args.model)
    row = None
    for candidate in dataset['test']:
        text = candidate['text']
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        if len(tokenizer(text, truncation=False)['input_ids']) >= MAX_LENGTH:
            row = candidate
            break
    if row is None:
        raise ValueError('Expected an actual original row reaching the original 32768 cap.')
    data = Dataset.from_list([row])
    left, right = module.get_response_template_ids(tokenizer)
    collator = module.DataCollatorForCompletionOnlyLM(response_template=left,
                    instruction_template=right, tokenizer=tokenizer)
    trainer_args = module.SFTConfig(output_dir=str(output / 'gpu_sanity_trainer'),
                per_device_eval_batch_size=1, report_to='none', dataset_text_field='text',
                max_length=MAX_LENGTH, dataset_num_proc=1, eval_accumulation_steps=1)
    trainer = module.SFTTrainer(model=model, processing_class=tokenizer, args=trainer_args,
                train_dataset=data, eval_dataset=data, data_collator=collator)
    prepared_row = trainer.eval_dataset[0]
    if len(prepared_row['input_ids']) != MAX_LENGTH:
        raise ValueError('The GPU witness must use the complete original 32768 cap.')
    labels = collator([{'input_ids': prepared_row['input_ids']}])['labels']
    positions, _ = active_positions(labels)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    metrics = trainer.evaluate()
    if not math.isfinite(metrics['eval_loss']):
        raise ValueError('Longest-context selected-response CE must be finite.')
    summary = {**configuration, 'status': 'gpu_sanity_passed', 'full_evaluation_complete': False,
               'context_tokens': MAX_LENGTH, 'supervised_response_tokens': positions.numel(),
               'metrics': metrics, 'peak_process_allocated_bytes': {
                   str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}}
    save(output / 'summary.json', summary)
    save(output / 'status.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(SOURCES))
    parser.add_argument('--model', choices=['qwentaur14b-r64', 'base14b'])
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-devices', help='Physical CUDA indices, e.g. 0,1; sets CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--activation-reserve-gib', type=float, default=4.0)
    parser.add_argument('--gpu-max-memory', help='Optional JSON logical-GPU weight caps in GiB, e.g. {"0":16,"1":18}.')
    parser.add_argument('--ce-chunk-size', type=int, default=64)
    parser.add_argument('--cpu-witness', action='store_true')
    parser.add_argument('--gpu-sanity', action='store_true', help='Only the separate actual 32768-context GPU witness.')
    args = parser.parse_args()
    if args.ce_chunk_size < 1 or args.activation_reserve_gib <= 0:
        parser.error('Positive CE chunk size and activation reserve are required.')
    if not args.cpu_witness and (not args.task or not args.model or not args.prepared_dir or not args.gpu_devices):
        parser.error('Full jobs require --task --model --prepared-dir --gpu-devices.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('All output must stay within this project.')
    if args.prepared_dir:
        args.prepared_dir = args.prepared_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    if args.cpu_witness:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    save(output / 'status.json', {'status': 'preparing', 'task': args.task, 'model': args.model})
    try:
        if args.cpu_witness:
            cpu_witness(output)
        else:
            evaluate(args, output)
    except Exception as exc:
        save(output / 'status.json', {'status': 'failed', 'task': args.task, 'model': args.model, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
