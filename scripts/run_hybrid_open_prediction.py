#!/usr/bin/env python3
"""Export and score frozen typed-choice models on all original Hybrid CV rows.

The author X order, 20 subjects, eight validation folds and 163,135 held-out
actions are preserved. Current decision statistics are rendered as a new text
interface; this is not the study's original on-screen stimulus or raw dialogue.
"""
import argparse
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.paper_hybrid_predictive import ACTIONS, QUESTION, X_COLUMNS, state_from_author_x
from src.paper_horizon import load_agent

DATA = ROOT / 'external/HybridModellingProject/Data/fit/data'
FOLDS = ROOT / 'data/hybrid_paper_matched/fold_assignments.csv'
EXPECTED_ROWS = 163135
EXPECTED_FOLDS = 160


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def fold_assignments():
    assignments = {subject: {} for subject in range(1, 21)}
    with FOLDS.open(newline='') as handle:
        for row in csv.DictReader(handle):
            subject = int(row['subject'])
            index = int(row['author_X_row_1based'])
            fold = int(row['validation_fold_1based'])
            if subject not in assignments or index in assignments[subject] or not 1 <= fold <= 8:
                raise ValueError('Invalid or duplicate published fold assignment.')
            assignments[subject][index] = fold
    if sum(map(len, assignments.values())) != EXPECTED_ROWS:
        raise ValueError('Incomplete published fold assignments.')
    return assignments


def export(output):
    if output.exists():
        raise ValueError(f'New output directory required: {output}')
    output.mkdir(parents=True)
    assignment = fold_assignments()
    fold_counts = {(s, k): 0 for s in range(1, 21) for k in range(1, 9)}
    target_fields = ['id', 'subject', 'session', 'trial', 'author_X_row_1based',
                     'validation_fold_1based', 'observed_action']
    total = 0
    with (output / 'requests.jsonl').open('w') as requests, \
         (output / 'targets.csv').open('w', newline='') as target_file:
        targets = csv.DictWriter(target_file, fieldnames=target_fields)
        targets.writeheader()
        for subject in range(1, 21):
            subject_dir = DATA / str(subject)
            with (subject_dir / 'X.csv').open(newline='') as x_file, \
                 (subject_dir / 'y.csv').open(newline='') as y_file:
                x_reader, y_reader = csv.DictReader(x_file), csv.DictReader(y_file)
                if x_reader.fieldnames != X_COLUMNS or y_reader.fieldnames != ACTIONS:
                    raise ValueError(f'Unexpected original X/y columns for subject {subject}.')
                subject_rows = 0
                for index, pair in enumerate(zip_longest(x_reader, y_reader), 1):
                    x, y = pair
                    if x is None or y is None:
                        raise ValueError(f'Original X/y row count differs for subject {subject}.')
                    if int(x['subject']) != subject:
                        raise ValueError('Published X row has wrong subject.')
                    chosen = [key for key in ACTIONS if y[key].lower() == 'true']
                    if len(chosen) != 1 or any(y[key].lower() not in ('true', 'false') for key in ACTIONS):
                        raise ValueError('Original y row is not a four-action one-hot label.')
                    fold = assignment[subject].get(index)
                    if fold is None:
                        raise ValueError('An original X row has no published validation fold.')
                    identity = f'{subject}:{index}'
                    request = {'id': identity, 'state': state_from_author_x(x),
                               'question': QUESTION, 'output_keys': ACTIONS}
                    requests.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n')
                    targets.writerow({'id': identity, 'subject': subject,
                                      'session': x['session'], 'trial': x['trial'],
                                      'author_X_row_1based': index,
                                      'validation_fold_1based': fold,
                                      'observed_action': chosen[0]})
                    fold_counts[(subject, fold)] += 1
                    total += 1
                    subject_rows += 1
                if subject_rows != len(assignment[subject]):
                    raise ValueError('Subject row count differs from published fold assignments.')
    if total != EXPECTED_ROWS or len(fold_counts) != EXPECTED_FOLDS or min(fold_counts.values()) < 1:
        raise ValueError('Original full held-out task was not exported.')
    summary = {
        'status': 'complete', 'mode': 'export', 'requests': total,
        'subjects': 20, 'validation_folds': EXPECTED_FOLDS,
        'all_rows_heldout_once': True,
        'source_X_y': str(DATA), 'source_folds': str(FOLDS),
        'source_methods': [
            'external/HybridModellingProject/Data/model/make_data.jl',
            'external/HybridModellingProject/Data/fit/make-data.jl',
            'external/HybridModellingProject/Data/list/scr/get data_list.R',
        ],
        'model_input': 'Only original current-decision n/r/mu, cursor/visit indicators, blocked '
                       'initial count and event-step t. Identifiers are target metadata, never state.',
        'unvisited_side': 'Original n value is the initial green reveal count, not already observed '
                          'black dots; current X has r=0 and mu=0.5. These describe the fitted '
                          'model row, not everything a participant may recall from phase one.',
        'excluded': ['action/y', 'future mouse position', 'future feedback', 'hidden ups',
                     'outcome/accuracy', 'fitted teacher VoI', 'subject/session/trial identifiers'],
        'interface': 'New numeric-to-text typed-choice adaptation; not the original raw task screen.',
        'action_order': ACTIONS,
    }
    save_json(output / 'summary.json', summary)
    print(json.dumps({'status': 'complete', 'mode': 'export', 'requests': total,
                      'validation_folds': EXPECTED_FOLDS}), flush=True)


class UniformChoice:
    def choice_probabilities(self, state, question, output_keys):
        return [1.0 / len(output_keys)] * len(output_keys)


def finite_or_null(value):
    return value if math.isfinite(value) else None


def request_chunks(requests, targets, batch_size):
    """Stream aligned original rows without regrouping folds or changing inputs."""
    chunk = []
    for request_line, target in zip_longest(requests, targets):
        if request_line is None or target is None:
            raise ValueError('Requests and targets have different row counts.')
        request = json.loads(request_line)
        if request['id'] != target['id'] or request['output_keys'] != ACTIONS:
            raise ValueError('Published target and decision request are misaligned.')
        chunk.append((request, target))
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def predict_chunk(model, chunk, batch_size):
    if batch_size == 1:
        request, _ = chunk[0]
        began = time.monotonic()
        probability = model.choice_probabilities(
            request['state'], request['question'], request['output_keys'])
        elapsed = time.monotonic() - began
        return [{
            'probabilities': probability,
            'native_choice': getattr(model, 'last_native_choice', None),
            'input_tokens': getattr(model, 'last_input_tokens', None),
            'request_index': 0, 'batch_id': 0, 'batch_size': 1,
            'batch_elapsed_seconds': elapsed, 'batch_amortized_seconds': elapsed,
            'inference_seconds': elapsed,
        }]
    from src.batched_choices import ChoiceRequest, choice_probabilities_batch

    prepared = [ChoiceRequest(request['state'], request['question'], request['output_keys'])
                for request, _ in chunk]
    results = choice_probabilities_batch(model, prepared, batch_size=batch_size)
    if len(results) != len(chunk):
        raise ValueError('Batch result count differs from original request count.')
    if any(row['request_index'] != index for index, row in enumerate(results)):
        raise ValueError('Batch results are not aligned with original request order.')
    return results


def run(output, export_dir, adapter_file, device, uniform, batch_size=1):
    if batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    if uniform and batch_size != 1:
        raise ValueError('Uniform sanity uses batch_size=1; batching requires a supported model adapter.')
    if output.exists():
        raise ValueError(f'New output directory required: {output}')
    source = json.loads((export_dir / 'summary.json').read_text())
    if source.get('status') != 'complete' or source.get('requests') != EXPECTED_ROWS:
        raise ValueError('The complete original held-out export is required.')
    output.mkdir(parents=True)
    started = time.monotonic()
    status = {'status': 'loading_model', 'mode': 'uniform_sanity' if uniform else 'frozen_prediction',
              'expected_rows': EXPECTED_ROWS, 'scored_rows': 0,
              'adapter_file': str(adapter_file.resolve()) if adapter_file else None,
              'device': device, 'batch_size_requested': batch_size,
              'completed_batches': 0,
              'timing_scope': ('Serial request timing on a shared device; not exclusive latency.'
                               if batch_size == 1 else
                               'Shared batch timing; amortized seconds are throughput accounting, '
                               'not isolated request latency. inference_seconds is empty.')}
    save_json(output / 'status.json', status)
    model = UniformChoice() if uniform else load_agent(adapter_file.resolve(), device)
    save_json(output / 'model_metadata.json', getattr(model, 'metadata', None))
    status['status'] = 'running'
    save_json(output / 'status.json', status)
    fold_sums = {(subject, fold): [0, 0.0, 0] for subject in range(1, 21) for fold in range(1, 9)}
    fields = ['id', 'subject', 'validation_fold_1based', 'author_X_row_1based',
              'observed_action', 'p_sample_left', 'p_sample_right', 'p_select_left',
              'p_select_right', 'action_nll_unsmoothed', 'native_choice', 'input_tokens',
              'inference_seconds', 'batch_id', 'batch_size', 'batch_elapsed_seconds',
              'batch_amortized_seconds']
    measured_seconds = 0.0
    current = None
    try:
        with (export_dir / 'requests.jsonl').open() as requests, \
             (export_dir / 'targets.csv').open(newline='') as target_file, \
             (output / 'heldout_predictions.csv').open('w', newline='') as prediction_file:
            writer = csv.DictWriter(prediction_file, fieldnames=fields)
            writer.writeheader()
            for chunk in request_chunks(requests, csv.DictReader(target_file), batch_size):
                current = {'batch_first_id': chunk[0][0]['id'], 'batch_last_id': chunk[-1][0]['id']}
                results = predict_chunk(model, chunk, batch_size)
                batch_offset = status['completed_batches']
                batch_times = {row['batch_id']: row['batch_elapsed_seconds'] for row in results}
                measured_seconds += sum(batch_times.values())
                for (request, target), result in zip(chunk, results):
                    current = {'id': request['id'], 'subject': int(target['subject']),
                               'validation_fold': int(target['validation_fold_1based'])}
                    p = np.asarray(result['probabilities'], dtype=float)
                    if (p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any()
                            or abs(float(p.sum()) - 1.0) > 1e-8):
                        raise ValueError('Model must return four normalized finite probabilities in original action order.')
                    choice = target['observed_action']
                    if choice not in ACTIONS:
                        raise ValueError('Target action is outside the original four labels.')
                    observed_p = float(p[ACTIONS.index(choice)])
                    nll = -math.log(observed_p) if observed_p > 0 else math.inf
                    key = (current['subject'], current['validation_fold'])
                    bucket = fold_sums[key]
                    bucket[0] += 1
                    bucket[1] += nll
                    bucket[2] += observed_p == 0
                    native = result['native_choice']
                    if native is not None and native not in ACTIONS:
                        raise ValueError('Adapter native choice is not a legal original action.')
                    writer.writerow({
                        'id': request['id'], 'subject': target['subject'],
                        'validation_fold_1based': target['validation_fold_1based'],
                        'author_X_row_1based': target['author_X_row_1based'],
                        'observed_action': choice,
                        **{f'p_{key}': float(value) for key, value in zip(ACTIONS, p)},
                        'action_nll_unsmoothed': nll, 'native_choice': native,
                        'input_tokens': result['input_tokens'],
                        'inference_seconds': result.get('inference_seconds') if batch_size == 1 else None,
                        'batch_id': batch_offset + result['batch_id'],
                        'batch_size': result['batch_size'],
                        'batch_elapsed_seconds': result['batch_elapsed_seconds'],
                        'batch_amortized_seconds': result['batch_amortized_seconds'],
                    })
                    status['scored_rows'] += 1
                status['completed_batches'] += len(batch_times)
                previous_rows = status['scored_rows'] - len(chunk)
                if status['scored_rows'] // 1000 > previous_rows // 1000:
                    save_json(output / 'status.json', {**status, 'last_scored': current})
                if status['scored_rows'] // 10000 > previous_rows // 10000:
                    print(json.dumps({'status': 'running', 'scored_rows': status['scored_rows'],
                                      'expected_rows': EXPECTED_ROWS}), flush=True)
        if status['scored_rows'] != EXPECTED_ROWS or any(n == 0 for n, _, _ in fold_sums.values()):
            raise ValueError('The full 20×8 held-out evaluation did not complete.')
        subject_means = []
        for subject in range(1, 21):
            losses = [fold_sums[(subject, fold)][1] / fold_sums[(subject, fold)][0]
                      for fold in range(1, 9)]
            subject_means.append(sum(losses) / 8)
        total_nll = sum(entry[1] for entry in fold_sums.values())
        zero_count = sum(entry[2] for entry in fold_sums.values())
        summary = {
            **status, 'status': 'sanity_complete' if uniform else 'complete',
            'subjects': 20, 'validation_folds': EXPECTED_FOLDS,
            'heldout_actions': status['scored_rows'],
            'mean_test_loss_equal_subject_fold_mean': finite_or_null(sum(subject_means) / 20),
            'pooled_action_nll': finite_or_null(total_nll / status['scored_rows']),
            'zero_observed_probability_count': zero_count,
            'infinite_loss_rule': 'Unsmoothed -log(p_observed); any zero makes aggregate +infinity, '
                                  'represented as null in JSON and inf in row CSV. No rows removed.',
            'action_order': ACTIONS, 'export_source': str(export_dir),
            'model_metadata': getattr(model, 'metadata', None),
            'seconds': time.monotonic() - started,
            'measured_inference_seconds_total': measured_seconds,
            'measured_inference_rows_per_second': status['scored_rows'] / measured_seconds,
            'scope_note': 'All 163135 author rows scored once on their original eight-fold '
                          'validation assignment. Frozen models use no fold fitting.',
            'interface_note': 'Numerical current-state text, not original human task stimulus.',
        }
        save_json(output / 'summary.json', summary)
        save_json(output / 'status.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'model_metadata'}), flush=True)
    except Exception as exc:
        save_json(output / 'status.json', {**status, 'status': 'failed', 'at': current,
                                           'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['export', 'predict'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--adapter-file', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--uniform-sanity', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Independent held-out rows per batch; default preserves serial inference.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep result output in this project.')
    if args.mode == 'export':
        export(output)
    else:
        if args.export_dir is None or (args.adapter_file is None and not args.uniform_sanity):
            parser.error('Predict mode requires --export-dir and --adapter-file unless --uniform-sanity.')
        run(output, args.export_dir.resolve(), args.adapter_file, args.device, args.uniform_sanity,
            args.batch_size)


if __name__ == '__main__':
    main()
