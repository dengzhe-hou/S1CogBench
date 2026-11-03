"""Run full original WCST test: frozen models, human history and own-action history.

Original decoder prompts/policies retained. Laya and the added 4B choice adapters
use the original Centaur visible state plus the shared typed WCST question, each
adapter's native choice and its existing calibration.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/laya'))
from src.wcst_task import Dialogue, LETTERS, QUESTION, task_metrics, test_data

MODEL_DIRS = {
    'laya': ROOT / 'models/laya-english',
    'llama8b': ROOT / 'models/llama-3.1-8b-instruct',
    'minitaur8b': ROOT / 'models/minitaur8b',
    'intern_decision_4b': ROOT / 'models/intern-decision-4b',
    'qwen35_4b': ROOT / 'models/qwen3.5-4b',
}
CHOICE_ADAPTERS = {
    'intern_decision_4b': 'examples.intern_decision_fullcontext_choice_agent',
    'qwen35_4b': 'examples.qwen35_4b_choice_agent',
}
ORIGINAL_FULL_VOCAB_MODELS = {'llama8b', 'minitaur8b'}


def predictive_metrics(records, original_full_vocab):
    result = {'human_nll_legal': float(np.mean([
        -np.log(r['legal_probabilities'][r['human_key'] - 1] + 1e-12) for r in records]))}
    if original_full_vocab:
        result['human_nll_original_full_vocab'] = float(np.mean([
            -r['original_logprobabilities'][r['human_key'] - 1] for r in records]))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Model:
    def __init__(self, name, device):
        self.name, self.device = name, torch.device(device)
        self.tokenizer = None
        self.has_original_full_vocab = name in ORIGINAL_FULL_VOCAB_MODELS
        if name == 'laya':
            import laya
            self.agent = laya.load(str(MODEL_DIRS[name]), device=device)
            self.agent.cfg['max_len'] = 8192
            self.model = self.agent.model
            self.limit = self.model.encoder.config.max_position_embeddings
            self.metadata = {'interface': 'Original Centaur visible state plus Laya typed choice question',
                             'policy': 'SDK native argmax; no external sampling',
                             'probabilities': 'SDK rounded four-decimal probabilities retained; normalized for legal NLL',
                             'temperature': self.agent.temperature,
                             'temperature_by_options': self.agent.temperature_by_options,
                             'max_len': 8192, 'head_max_len': self.agent.cfg['head_max_len']}
        elif name in CHOICE_ADAPTERS:
            self.agent = importlib.import_module(CHOICE_ADAPTERS[name]).create_agent(device)
            if name == 'intern_decision_4b':
                self.model = self.agent.engine.backend.model
                self.limit = self.agent.engine.backend.max_length
            else:
                self.model = self.agent.model
                self.limit = self.agent.max_context_tokens
            self.metadata = dict(self.agent.metadata)
            self.metadata.update({
                'adapter': CHOICE_ADAPTERS[name],
                'task_interface': 'Original Centaur visible state plus the same typed WCST choice question used by Laya',
                'predictive_policy': 'Adapter native legal-choice argmax; no external sampling',
                'generative_policy': 'Adapter native legal-choice argmax; no external sampling',
                'nll_scope': 'Legal-choice conditional NLL only; not original full-vocabulary token NLL',
            })
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRS[name], local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIRS[name], local_files_only=True, torch_dtype=torch.bfloat16).to(self.device).eval()
            pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer,
                            pad_token_id=0 if name == 'minitaur8b' else self.tokenizer.eos_token_id,
                            do_sample=True, temperature=1.0, max_new_tokens=1)
            self.generation_config = pipe.generation_config
            self.generation_config.do_sample = True
            self.generation_config.temperature = 1.0
            self.generation_config.max_new_tokens = 1
            self.generation_config.pad_token_id = 0 if name == 'minitaur8b' else self.tokenizer.eos_token_id
            self.letter_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
            assert all(len(ids) == 1 for ids in self.letter_ids)
            self.letter_ids = [ids[0] for ids in self.letter_ids]
            self.limit = self.model.config.max_position_embeddings
            self.metadata = {'interface': 'Official Llama chat / Centaur marker prompt functions',
                             'predictive_policy': 'Original full-vocabulary argmax',
                             'generative_policy': 'Original one-token sampling at temperature 1; checkpoint/pipeline defaults retained',
                             'effective_generation_config': self.generation_config.to_dict(),
                             'probabilities': 'Original full-vocabulary p; additional legal-ABCD conditional p separately',
                             'adaptation': 'Current cached 8B weights; not a rerun of the 70B model'}
            del pipe
        self.metadata.update({'model_directory': str(MODEL_DIRS[name]),
                              'original_full_vocab_nll_available': self.has_original_full_vocab,
                              'legal_nll_definition': 'Mean -log(p_human + 1e-12), preserving the original WCST legal-NLL reporting convention',
                              'parameter_dtypes': sorted({str(p.dtype) for p in self.model.parameters()}),
                              'history': 'Full original history, reset each participant, no truncation or summary',
                              'training': 'Frozen; no task training or prompt selection',
                              'context_limit': int(self.limit), 'device': str(self.device)})

    def act(self, prompt, mode):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if self.name == 'laya':
            from laya.common import build_sequence
            internal = {'t': 'choice', 'ins': QUESTION['instructions'], 'crit': QUESTION['criteria']}
            ids, markers = build_sequence(self.agent.tok, prompt, internal, 1000000,
                                          self.agent.cfg['head_max_len'])
            if len(ids) > self.limit:
                raise ValueError(f'Full history {len(ids)} exceeds context {self.limit}; no truncation.')
            assert len(markers) == 4
            output = self.agent.predict(prompt, {'action': QUESTION})
            if self.agent.device.type != 'cuda':
                raise RuntimeError('Laya SDK moved to CPU; do not silently change execution device.')
            answer = output['answers']['action']
            raw = np.array([answer['probabilities'][c] for c in LETTERS], dtype=float)
            p = raw / raw.sum()
            letter = answer['choice']
            result = {'key': LETTERS.index(letter) + 1, 'letter': letter,
                      'legal_probabilities': p.tolist(), 'sdk_raw': output,
                      'input_tokens': len(ids), 'allowed_token_mass': None,
                      'full_vocab_probabilities': None}
        elif self.name in CHOICE_ADAPTERS:
            p = self.agent.choice_probabilities(prompt, QUESTION, list(LETTERS))
            letter = self.agent.last_native_choice
            if letter not in LETTERS:
                raise ValueError(f'Native adapter returned an illegal WCST choice: {letter!r}')
            result = {'key': LETTERS.index(letter) + 1, 'letter': letter,
                      'legal_probabilities': p.tolist(),
                      'input_tokens': self.agent.last_input_tokens,
                      'allowed_token_mass': getattr(self.agent, 'last_allowed_token_mass', None),
                      'full_vocab_probabilities': None,
                      'label_mapping': self.agent.last_label_mapping,
                      'native_choice': letter}
            if self.name == 'intern_decision_4b':
                result.update(sdk_raw=self.agent.last_response,
                              raw_legal_probabilities=self.agent.last_raw_probabilities)
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            n = inputs.input_ids.shape[1]
            if n > self.limit:
                raise ValueError(f'Full history {n} exceeds context {self.limit}; no truncation.')
            with torch.inference_mode():
                if mode == 'predictive':
                    logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
                    token = int(torch.softmax(logits, -1).argmax())
                    letter = next((c for c, tid in zip(LETTERS, self.letter_ids) if tid == token), '?')
                    raw_text = self.tokenizer.decode([token], skip_special_tokens=True)
                else:
                    generated = self.model.generate(
                        **inputs, generation_config=self.generation_config, use_cache=False,
                        return_dict_in_generate=True, output_logits=True)
                    logits = generated.logits[0][0]
                    token = int(generated.sequences[0, -1])
                    raw_text = self.tokenizer.decode([token], skip_special_tokens=True)
                    if self.name == 'llama8b':
                        reply = raw_text.strip()
                        letter = reply[0].upper() if reply else '?'
                    else:
                        # Source takes the last character of full generated_text.
                        letter = (prompt + raw_text)[-1].upper()
            vocab_probs = torch.softmax(logits, -1)
            original_ll = torch.log(vocab_probs[self.letter_ids] + 1e-8).double().cpu().numpy()
            full = vocab_probs[self.letter_ids].double().cpu().numpy()
            p = torch.softmax(logits[self.letter_ids].double(), -1).cpu().numpy()
            result = {'key': LETTERS.index(letter) + 1 if letter in LETTERS else None,
                      'letter': letter, 'generated_token': token, 'raw_text': raw_text,
                      'legal_probabilities': p.tolist(), 'input_tokens': n,
                      'allowed_token_mass': float(full.sum()),
                      'full_vocab_probabilities': full.tolist(),
                      'original_logprobabilities': original_ll.tolist()}
        p = np.asarray(result['legal_probabilities'])
        assert np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1) < 1e-6
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result['seconds_shared_gpu'] = time.perf_counter() - start
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_DIRS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true', help='Both modes, first and longest state only; not a benchmark score')
    parser.add_argument('--smoke-subject', type=int, help='Use the CPU-checked longest participant for smoke')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'metadata.json').exists():
        raise FileExistsError('Use a fresh attempt directory; previous results are retained.')
    torch.set_num_threads(2)
    import transformers
    data = test_data()
    if args.smoke_subject is not None and (not args.smoke or args.smoke_subject not in data):
        parser.error('--smoke-subject requires --smoke and an original test participant ID')
    model = Model(args.model, args.device)
    torch.set_num_threads(2)
    meta = {'model': args.model, 'pid': os.getpid(), 'smoke_only': args.smoke,
            'participants': len(data), 'trials_per_mode': sum(map(len, data.values())),
            'seed': '10000 + original participant ID, reset independently in each mode',
            'split': 'Original SL ordered first20% participants, exactly75; no resplit',
            'environment': 'Original stimulus/hidden-rule timeline; feedback uses executed action == ground_key',
            'timing': 'Shared GPU timing, not human reaction time or exclusive deployment benchmark',
            'torch': torch.__version__, 'transformers': transformers.__version__, **model.metadata}
    if args.smoke:
        meta['smoke_history'] = 'First and terminal input of the selected participant; unscored intervening inputs replay human actions. This is not a full generative trajectory.'
        meta['smoke_subject'] = (args.smoke_subject if args.smoke_subject is not None
                                 else max(data, key=lambda s: len(data[s])))
    write_json(args.output / 'metadata.json', meta)
    completed, failures = [], []
    family = 'llama' if args.model == 'llama8b' else 'centaur'
    for mode in ['predictive', 'generative']:
        mode_dir = args.output / mode
        mode_dir.mkdir()
        for sid, rows in data.items():
            if args.smoke and sid != meta['smoke_subject']:
                continue
            random.seed(10000 + sid)
            np.random.seed(10000 + sid)
            torch.manual_seed(10000 + sid)
            torch.cuda.manual_seed_all(10000 + sid)
            dialogue = Dialogue(family, mode)
            records = []
            try:
                with (mode_dir / f'{sid}_trials.jsonl').open('w') as out:
                    for i, row in enumerate(rows):
                        if args.smoke and i not in (0, len(rows) - 1):
                            dialogue.update(row, row['user_key'])
                            continue
                        prompt = dialogue.prompt(row, model.tokenizer)
                        answer = model.act(prompt, mode)
                        actual = row['user_key'] if mode == 'predictive' else answer['key']
                        record = {'subject_id': sid, 'trial': i, 'source_step': int(row['step']),
                                  'human_key': int(row['user_key']), 'ground_key': int(row['ground_key']),
                                  'ground_rule': int(row['ground_rule']), 'stimulus': list(row['stimulus']),
                                  'executed_key': int(actual) if actual is not None else None,
                                  'feedback': 'REPEAT' if actual == row['ground_key'] else 'SWITCH', **answer}
                        out.write(json.dumps(record, allow_nan=False) + '\n')
                        out.flush()
                        records.append(record)
                        write_json(args.output / 'progress.json', {
                            'status': 'running', 'mode': mode, 'subject_id': sid, 'trial': i + 1,
                            'completed_participant_modes': len(completed), 'required_participant_modes': 150,
                            'completed_actions': sum(x['trials'] for x in completed) + len(records),
                            'failed_participant_modes': failures})
                        if args.smoke:
                            (mode_dir / f'{sid}_{i}_prompt.txt').write_text(prompt)
                            if args.model in CHOICE_ADAPTERS:
                                (mode_dir / f'{sid}_{i}_adapter_prompt.txt').write_text(model.agent.last_prompt)
                        dialogue.update(row, actual, answer['letter'] if mode == 'generative' else None)
                summary = {'subject_id': sid, 'mode': mode, 'trials': len(records),
                           'invalid_outputs': sum(r['key'] is None for r in records),
                           'max_input_tokens': max(r['input_tokens'] for r in records),
                           'seconds_shared_gpu': sum(r['seconds_shared_gpu'] for r in records)}
                if not args.smoke:
                    summary.update(task_metrics(rows, [r['key'] for r in records]))
                    if mode == 'predictive':
                        summary.update(predictive_metrics(records, model.has_original_full_vocab))
                write_json(mode_dir / f'{sid}_summary.json', summary)
                completed.append(summary)
                print(json.dumps(summary), flush=True)
            except (ValueError, RuntimeError, KeyError) as exc:
                failure = {'mode': mode, 'subject_id': sid, 'completed_trials': len(records),
                           'error': repr(exc), 'traceback': traceback.format_exc()}
                failures.append(failure)
                write_json(mode_dir / f'{sid}_failure.json', failure)
                print(json.dumps(failure), flush=True)
                # An interface failure leaves this participant incomplete; retain all others.
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
    status = 'smoke_complete' if args.smoke and not failures else ('complete' if len(completed) == 150 and not failures else 'incomplete')
    result = {'status': status, 'completed_participant_modes': len(completed),
              'completed_actions': sum(x['trials'] for x in completed), 'failures': failures,
              'per_participant': completed}
    if model.device.type == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(model.device)
        result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(model.device)
    if status == 'complete':
        result['metrics'] = {}
        for mode in ['predictive', 'generative']:
            parts = [x for x in completed if x['mode'] == mode]
            fields = ['accuracy', 'perseveration_error', 'set_loss_error']
            if mode == 'predictive':
                fields += ['human_nll_legal'] + (['human_nll_original_full_vocab'] if model.has_original_full_vocab else [])
            result['metrics'][mode] = {f: {'participant_mean': float(np.mean([x[f] for x in parts])),
                'trial_weighted_mean': float(np.average([x[f] for x in parts], weights=[x['trials'] for x in parts]))} for f in fields}
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'progress.json', {k: v for k, v in result.items() if k != 'per_participant'})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
