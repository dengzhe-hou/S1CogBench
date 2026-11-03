"""P3 original prompt families, frozen parameters, common legal readouts.

No neural inference occurs on import. Source tasks and fixed splits are read from
--root, normally the already-deployed probes_20261028/root.
"""
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import types

MODEL_IDS = ['laya', 'intern_decision_4b', 'llama8b', 'minitaur8b', 'qwen35_4b']
KEYS = ['participant_id', 'block', 'game', 'trial_num']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


def configure_root(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'external/laya'))
    return root


def load_horizon(root, family, pure_cpu=False):
    """Load unchanged original source functions; avoid the obsolete login helper."""
    source = Path(root) / f'external/paper_matched_centaur_evaluation/horizon/horizon_{family}_one.py'
    if pure_cpu:
        import numpy as np
        import pandas as pd
        # Only functions are executed for a tiny deterministic data/loop check.
        tree = ast.parse(source.read_text())
        namespace = {'np': np, 'pd': pd, 'random': random,
                     'torch': types.SimpleNamespace(manual_seed=lambda seed: None),
                     'transformers': types.SimpleNamespace(pipeline=object, set_seed=lambda seed: None)}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'main']
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        return types.SimpleNamespace(**namespace)
    helper = types.ModuleType('get_models')
    previous = sys.modules.get('get_models')
    sys.modules['get_models'] = helper
    try:
        spec = importlib.util.spec_from_file_location('p3_official_horizon_' + family, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['get_models']
        else:
            sys.modules['get_models'] = previous
    return module


def choose(probabilities, keys, readout):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(keys),) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 1e-8:
        raise ValueError('Invalid legal-action probability distribution.')
    if readout == 'argmax':
        return keys[int(p.argmax())]
    if readout == 'sample':
        return str(np.random.choice(keys, p=p))
    raise ValueError('Readout must be argmax or sample.')


class OriginalPromptModel:
    """Retain each task/model's original prompt interface and forward precision."""
    def __init__(self, name, root, device):
        import torch
        self.name, self.root, self.device = name, configure_root(root), torch.device(device)
        self.tokenizer = None
        self.task = None
        if name == 'laya':
            import laya
            self.agent = laya.load(str(self.root / 'models/laya-english'), device=device)
            self.sdk_default_forward_dtype = self.agent.dtype
            self.agent.cfg['max_len'] = 8192
            self.model = self.agent.model
        elif name in ['intern_decision_4b', 'qwen35_4b']:
            module = {'intern_decision_4b': 'examples.intern_decision_fullcontext_choice_agent',
                      'qwen35_4b': 'examples.qwen35_4b_choice_agent'}[name]
            self.agent = importlib.import_module(module).create_agent(device)
            self.model = (self.agent.engine.backend.model if name == 'intern_decision_4b'
                          else self.agent.model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            directories = {'llama8b': self.root / 'models/llama-3.1-8b-instruct',
                           'minitaur8b': Path(os.environ.get('S1COGBENCH_MINITAUR_MODEL_DIR', str(self.root / 'models/minitaur8b')))}
            self.model_directory = directories[name]
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_directory,
                local_files_only=True, torch_dtype=torch.bfloat16).to(self.device).eval()
        actual_kernels = sorted({layer.chunk_gated_delta_rule.__module__ for layer in self.model.modules()
                                 if hasattr(layer, 'chunk_gated_delta_rule')})
        actual_norms = sorted({type(layer.norm).__module__+'.'+type(layer.norm).__name__ for layer in self.model.modules()
                               if hasattr(layer, 'chunk_gated_delta_rule')})
        if name in ['intern_decision_4b', 'qwen35_4b']:
            if actual_kernels != ['fla.ops.gated_delta_rule.chunk'] or actual_norms != ['fla.modules.fused_norm_gate.FusedRMSNormGated']:
                raise RuntimeError('P3 must retain the archived FLA kernel/norm; do not silently switch to another implementation.')
        self.metadata = {'name': name, 'parameter_dtypes': sorted({str(p.dtype) for p in self.model.parameters()}),
                         'linear_attention_kernel_modules': actual_kernels,
                         'linear_attention_norm_classes': actual_norms,
                         'device': str(self.device), 'frozen': True,
                         'legal_readout': 'Temperature-one normalized legal probabilities; argmax or categorical sampling; ties use first legal key.',
                         'history': 'Complete source task history, no truncation or summaries.',
                         'model_adapter': getattr(getattr(self, 'agent', None), 'metadata', None)}

    def set_task(self, task):
        import torch
        self.task = task
        if self.name == 'laya':
            # Original Horizon explicitly used FP32; original WCST used the SDK default.
            self.agent.dtype = torch.float32 if task == 'horizon' else self.sdk_default_forward_dtype
            self.metadata['actual_laya_forward_dtype'] = str(self.agent.dtype)

    def probabilities(self, state, question, keys):
        import numpy as np
        import torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if self.name == 'laya':
            from laya.common import build_sequence
            internal = {'t': 'choice', 'ins': question['instructions'], 'crit': question['criteria']}
            ids, markers = build_sequence(self.agent.tok, state, internal, 1000000,
                                           self.agent.cfg['head_max_len'])
            limit = int(self.model.encoder.config.max_position_embeddings)
            if len(ids) > limit or len(markers) != len(keys):
                raise ValueError('Full source prompt does not fit Laya context/head; no truncation.')
            answer = self.agent.predict(state, {'action': question})['answers']['action']
            if self.agent.device != self.device:
                raise RuntimeError('Laya changed execution device.')
            raw = np.array([answer['probabilities'][key] for key in keys], dtype=float)
            if not np.isfinite(raw).all() or (raw < 0).any() or raw.sum() <= 0 or abs(raw.sum()-1) > len(raw)*.00005+1e-8:
                raise ValueError('Invalid rounded SDK probabilities.')
            p = raw/raw.sum()
            extra = {'raw_legal_probabilities': raw.tolist(), 'input_tokens': len(ids),
                     'native_choice': answer['choice'], 'actual_forward_dtype': str(self.agent.dtype)}
        elif self.name in ['intern_decision_4b', 'qwen35_4b']:
            p = np.asarray(self.agent.choice_probabilities(state, question, keys), dtype=float)
            extra = {'input_tokens': self.agent.last_input_tokens,
                     'native_choice': self.agent.last_native_choice,
                     'allowed_token_mass': getattr(self.agent, 'last_allowed_token_mass', None)}
        else:
            # Direct original model-family prompt, no extra chat/typed wrapping.
            inputs = self.tokenizer(state, return_tensors='pt').to(self.device)
            n = int(inputs.input_ids.shape[1])
            if n > int(self.model.config.max_position_embeddings):
                raise ValueError('Complete source history exceeds context; no truncation.')
            ids = [self.tokenizer.encode(k, add_special_tokens=False) for k in keys]
            if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(keys):
                raise ValueError('Original legal keys must be distinct single tokens.')
            ids = [x[0] for x in ids]
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits[ids].double(), -1).cpu().numpy()
            extra = {'input_tokens': n, 'allowed_token_mass': float(torch.softmax(logits, -1)[ids].sum()),
                     'full_vocab_argmax_token': int(logits.argmax()),
                     'interface': 'Original prompt; legal conditioning replaces original full-vocabulary sampling.'}
        torch.cuda.synchronize(self.device)
        extra.update(inference_seconds=time.perf_counter()-started, legal_probabilities=p.tolist())
        choose(p, keys, 'argmax')  # Direct validation; no RNG consumption.
        return p, extra


def horizon_question(root):
    # Exact benchmark question function, without importing unrelated models.
    source = Path(root) / 'src/horizon.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'choice_question')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['choice_question'](labels=('H', 'I'))


def run_horizon(model, root, output, readout):
    import numpy as np
    import pandas as pd
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    family = 'llama' if model.name == 'llama8b' else 'centaur'
    official = load_horizon(root, family)
    timeline = pd.read_csv(Path(root)/'external/paper_matched_centaur_evaluation/horizon/data/in/timeline_structure.csv')
    if (timeline.participant_id.nunique(), len(timeline), int(timeline.type.eq('free').sum())) != (31,74400,34720):
        raise ValueError('Original Horizon full-sample count mismatch.')
    seeds = official.generate_seeds(num_seeds=32)
    question = horizon_question(root)
    model.set_task('horizon')
    write_json(output/'metadata.json', {'task': 'horizon', 'model': model.metadata, 'readout': readout,
        'prompt_family': family, 'participant_seeds': {str(int(p)): seeds[int(p)-1] for p in timeline.participant_id.unique()},
        'expected_rows': 74400, 'expected_decisions': 34720, 'participants': 31})
    frames=[]; records=[]
    with (output/'calls.jsonl').open('w') as stream:
        class Pipe:
            def __call__(self, prompt):
                p, extra = model.probabilities(prompt, question, ['H','I'])
                choice = choose(p, ['H','I'], readout)
                record={'id': len(records), 'choice': choice, **extra}
                records.append(record); stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
                return [{'generated_text': prompt+choice}]
        pipe=Pipe()
        for participant in timeline.participant_id.unique():
            part=timeline[timeline.participant_id.eq(participant)]
            result=official.simulate_participant_by_block(part,participant,seeds,None,None,pipe)
            # Original immutable trial identities, forced actions and reward routing.
            expected=part.rename(columns={'trial_num_block':'trial_num'}).set_index(KEYS).sort_index()
            actual=result.set_index(KEYS).sort_index()
            pd.testing.assert_index_equal(actual.index,expected.index)
            free=actual.is_free.astype(bool)
            np.testing.assert_array_equal(free,expected.type.eq('free'))
            np.testing.assert_array_equal(actual.loc[~free,'choice'],expected.loc[~free,'choice'])
            np.testing.assert_array_equal(actual.loc[~free,'reward'],expected.loc[~free,'reward'])
            routed=np.where(actual.choice.eq('H'),expected.reward_H,expected.reward_I)
            np.testing.assert_array_equal(actual.loc[free,'reward'],routed[free])
            if family == 'centaur':
                from verify_horizon import verify_rollout
                verify_rollout(part, result)
            else:
                for (block, game), game_rows in result.groupby(['block','game'],sort=False):
                    src_game = part[part.block.eq(block) & part.game.eq(game)].sort_values('trial_num_block')
                    forced = src_game[src_game.type.eq('forced')]
                    running_reward = forced.reward.sum()
                    previous_free = game_rows.iloc[:0]
                    for row in game_rows[game_rows.is_free.astype(bool)].sort_values('trial_num').itertuples():
                        expected_prompt = official.build_slot_prompt_llama(
                            current_trial=row.trial_num,
                            past_trials=official.format_past_trials(previous_free),
                            forced_choices=official.format_forced_trials(forced),
                            total_trials=len(src_game),game_number=game)
                        if row.prompt != expected_prompt:
                            raise ValueError('Original Llama prompt does not match its own executed history.')
                        running_reward += row.reward
                        if row.cumulative_reward != running_reward:
                            raise ValueError('Incorrect free-trial cumulative reward.')
                        previous_free = game_rows[game_rows.is_free.astype(bool) & game_rows.trial_num.le(row.trial_num)]
            result.to_csv(output/f'participant_{int(participant)}.csv',index=False)
            frames.append(result)
            write_json(output/'progress.json', {'status':'running','completed_participants':len(frames),'participants':31,
                       'completed_decisions':len(records),'expected_decisions':34720})
            print(json.dumps({'task':'horizon','readout':readout,'completed_participants':len(frames),'decisions':len(records)}),flush=True)
    full=pd.concat(frames,ignore_index=True); free=full[full.is_free.astype(bool)].copy()
    if len(full)!=74400 or len(free)!=34720 or len(records)!=34720: raise ValueError('Incomplete Horizon rollout.')
    free['optimal']=((free.choice.eq('H') & free.reward_mean_H.gt(free.reward_mean_I)) |
                     (free.choice.eq('I') & free.reward_mean_I.gt(free.reward_mean_H)))
    full.to_csv(output/'trials.csv',index=False)
    by_trial=free.groupby(['participant_id','horizon','trial_num']).optimal.mean()
    by_trial.groupby(['horizon','trial_num']).agg(['mean','sem']).to_csv(output/'optimal_choice_by_trial.csv')
    summary={'status':'complete','task':'horizon','readout':readout,'participants':31,'decisions':len(records),'rows':len(full),
             'mean_reward_free_additional_metric':float(free.reward.mean()),'mean_optimal_choice_rate':float(free.optimal.mean())}
    write_json(output/'summary.json',summary); write_json(output/'progress.json',summary)
    return summary


def run_wcst(model, root, output, readout):
    import numpy as np
    import torch
    from src.wcst_task import Dialogue, LETTERS, QUESTION, task_metrics, test_data
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    data=test_data(); family='llama' if model.name=='llama8b' else 'centaur'
    model.set_task('wcst')
    write_json(output/'metadata.json', {'task':'wcst','model':model.metadata,'readout':readout,'prompt_family':family,
        'seed':'10000 + original subject_id','participants':75,'expected_decisions':12754,
        'split':'Source ordered first20% participants; source stimuli/rule timeline fixed.'})
    summaries=[]; actions=0
    for sid,rows in data.items():
        random.seed(10000+sid); np.random.seed(10000+sid); torch.manual_seed(10000+sid); torch.cuda.manual_seed_all(10000+sid)
        dialogue=Dialogue(family,'generative'); records=[]
        with (output/f'{sid}_trials.jsonl').open('w') as stream:
            for i,row in enumerate(rows):
                prompt=dialogue.prompt(row,model.tokenizer)
                p,extra=model.probabilities(prompt,QUESTION,list(LETTERS))
                letter=choose(p,list(LETTERS),readout); key=LETTERS.index(letter)+1
                record={'subject_id':sid,'trial':i,'source_step':int(row['step']),
                        'human_key':int(row['user_key']),'ground_key':int(row['ground_key']),
                        'ground_rule':int(row['ground_rule']),'stimulus':list(row['stimulus']),
                        'key':key,'letter':letter,'executed_key':key,
                        'feedback':'REPEAT' if key==row['ground_key'] else 'SWITCH',**extra}
                stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); records.append(record)
                dialogue.update(row,key,letter)
        part={'subject_id':sid,'trials':len(records),**task_metrics(rows,[x['key'] for x in records]),
              'max_input_tokens':max(x['input_tokens'] for x in records)}
        summaries.append(part); actions+=len(records)
        write_json(output/f'{sid}_summary.json',part)
        write_json(output/'progress.json',{'status':'running','completed_participants':len(summaries),
                   'participants':75,'completed_decisions':actions,'expected_decisions':12754})
        print(json.dumps({'task':'wcst','readout':readout,'completed_participants':len(summaries),'decisions':actions}),flush=True)
    if len(summaries)!=75 or actions!=12754: raise ValueError('Incomplete WCST rollout.')
    summary={'status':'complete','task':'wcst','readout':readout,'participants':75,'decisions':actions,'per_participant':summaries,
             'metrics':{field:{'participant_mean':float(np.mean([x[field] for x in summaries])),
                        'trial_weighted_mean':float(np.average([x[field] for x in summaries],weights=[x['trials'] for x in summaries]))}
                        for field in ['accuracy','perseveration_error','set_loss_error']}}
    write_json(output/'summary.json',summary); write_json(output/'progress.json',{k:v for k,v in summary.items() if k!='per_participant'})
    return summary
