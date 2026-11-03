#!/usr/bin/env python3
"""Token-only length audit; keep all 6,720 published Horizon prompts intact."""
import ast
import importlib.util
import json
import os
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]


def functions_from_source(path, names):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Expected published prompt functions were not found.')
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    official = ROOT / 'external/paper_matched_centaur_evaluation/horizon'
    helpers = functions_from_source(official / 'predictive_horizon_centaur.py',
                                    ['build_game_intro', 'format_forced_trials', 'format_past_trials'])
    choice = functions_from_source(ROOT / 'src/horizon.py', ['choice_question'])['choice_question']
    spec = importlib.util.spec_from_file_location('laya_sequence_only', ROOT / 'external/laya/laya/common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    checkpoint = ROOT / 'models/laya-english'
    cfg = json.loads((checkpoint / 'rl_agent_config.json').read_text())
    encoder = json.loads((checkpoint / 'encoder/config.json').read_text())
    tok = AutoTokenizer.from_pretrained(checkpoint / 'tokenizer', local_files_only=True)
    external_question = choice(labels=('H', 'I'))
    q = {'t': external_question['type'], 'ins': external_question['instructions'],
         'crit': external_question['criteria']}
    head_ids, _ = common.build_sequence(tok, '', q, cfg['max_len'], cfg['head_max_len'])
    data = pd.read_csv(official / 'data/in/test_data.csv')
    records = []
    for (participant, block, game), game_df in data.groupby(['participant_id', 'block', 'game'], sort=True):
        game_df = game_df.sort_values('trial')
        intro = helpers['build_game_intro'](game_df, game)
        forced = helpers['format_forced_trials'](game_df[game_df.type.eq('forced')])
        for _, row in game_df[game_df.type.eq('free')].iterrows():
            previous = game_df[game_df.type.eq('free') & game_df.trial.lt(row.trial)]
            past = helpers['format_past_trials'](previous)
            prompt = str(intro + forced + past) + 'You press <<'
            if tok.mask_token in prompt:
                raise ValueError('An upstream prompt contains the SDK mask token.')
            raw = tok(prompt, add_special_tokens=False, truncation=False)['input_ids']
            required = len(head_ids) + len(raw)
            full, markers = common.build_sequence(tok, prompt, q, required, cfg['head_max_len'])
            configured, configured_markers = common.build_sequence(
                tok, prompt, q, cfg['max_len'], cfg['head_max_len'])
            if len(full) != required or len(markers) != 2:
                raise ValueError('SDK serialization does not match the complete state plus choice head.')
            records.append({'participant_id': int(participant), 'block': int(block), 'game': int(game),
                            'trial': int(row.trial), 'prompt_tokens': len(raw),
                            'sdk_required_tokens': required,
                            'sdk_default_truncates': full != configured or markers != configured_markers})
    if len(records) != 6720:
        raise ValueError(f'Expected 6,720 prompts, got {len(records)}.')

    def stats(key):
        values = np.array([r[key] for r in records])
        return {'min': int(values.min()), 'max': int(values.max()),
                'quantiles': {str(q): float(np.quantile(values, q)) for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]},
                'exceeds_sdk_configured_512': int((values > cfg['max_len']).sum()),
                'exceeds_encoder_8192': int((values > encoder['max_position_embeddings']).sum())}

    result = {
        'date': '2026-10-03', 'status': 'tokenizer_only_no_model_inference',
        'protocol': 'Namazova published full six-participant predictive test',
        'n_prompts': len(records), 'prompt_construction': 'unmodified published Centaur per-game string functions',
        'laya_sdk_max_len': cfg['max_len'], 'laya_sdk_head_max_len': cfg['head_max_len'],
        'tokenizer_model_max_length': tok.model_max_length,
        'encoder_max_position_embeddings': encoder['max_position_embeddings'],
        'audit_question_only_not_an_adopted_protocol': external_question,
        'choice_head_and_special_token_overhead': len(head_ids),
        'prompt_only': stats('prompt_tokens'), 'with_laya_sdk_choice_head': stats('sdk_required_tokens'),
        'default_sdk_truncation_count': sum(r['sdk_default_truncates'] for r in records),
        'longest_prompt_key': max(records, key=lambda r: r['sdk_required_tokens']),
        'notes': [
            '512 is the shipped Laya SDK serialization budget, not the tokenizer or encoder positional limit.',
            'SDK prepends type/instructions, option masks/options and separator tokens; all are counted.',
            'An API choice head is still an interface adaptation; length feasibility does not establish identical inference protocol.',
            'No prompt text or model weights were modified; no truncation or inference was performed.',
        ],
    }
    out = ROOT / 'data/paper_matched_sources/laya_horizon_prompt_lengths_20261003.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
