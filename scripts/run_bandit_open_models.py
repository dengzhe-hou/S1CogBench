#!/usr/bin/env python3
"""Closed-loop typed-choice evaluation on EVOLvE's public FullBench.

Raw History and Summary History are separate official MAB context conditions.
The 30 repetitions match this project's completed CPU controls and the paper's
count; public BenchBase itself defaults to 20 and uses different trajectory seeds.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/EVOLvE'))

import numpy as np
from banditbench import FullBench
from banditbench.agents.llm import LLMMABAgentRH, LLMMABAgentSH
from banditbench.sampling.sampler import get_trajectory_seeds

from src.paper_horizon import load_agent

N_TRAJECTORIES = 30
REPLAY_SEED_BASE = 20261009
SOURCE_LABEL = 'official_code_reproduction_not_paper_matched'


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def specifications(bench):
    specs = []
    for index, (core, verbal) in enumerate(zip(bench.envs, bench.verbal_envs)):
        specs.append({
            'condition_index': index,
            'condition_id': f'{index:02d}_{type(verbal.bandit_scenario).__name__}_{core.name}',
            'environment_name': core.name,
            'scenario': type(verbal.bandit_scenario).__name__,
            'reward_distribution': type(core).__name__,
            'num_arms': core.num_arms,
            'horizon': core.horizon,
            'initial_environment_seed': core.seed,
            'trajectory_seeds': [int(x) for x in get_trajectory_seeds(core.seed, N_TRAJECTORIES)],
            'action_names': list(verbal.action_names),
            'arm_expected_rewards_scoring_only': [float(core.expected_reward(i)) for i in range(core.num_arms)],
            'arm_params_scoring_only': core.arm_params,
        })
    if len(specs) != 16 or len({s['condition_id'] for s in specs}) != 16:
        raise ValueError('Expected all 16 uniquely identified FullBench conditions.')
    if sum(s['horizon'] for s in specs) * N_TRAJECTORIES != 312000:
        raise ValueError('The FullBench horizons have changed.')
    return specs


def visible_prompt(prompt_agent):
    """Mirror LLMMABAgentBase.act without calling its text-generation API."""
    return (prompt_agent.get_task_instruction()
            + prompt_agent.decision_context_start.format(len(prompt_agent.interaction_history))
            + prompt_agent.get_action_history()
            + prompt_agent.get_decision_query())


def run_episode(verbal_template, spec, repetition, model, history, action_rule, output):
    env = deepcopy(verbal_template)  # official LLM sampler also copies env per trajectory
    seed = spec['trajectory_seeds'][repetition]
    env.reset(seed=seed)
    prompt_class = LLMMABAgentRH if history == 'raw' else LLMMABAgentSH
    prompt_agent = prompt_class(env, history_context_len=1000)
    prompt_agent.reset()
    names = list(env.action_names)
    question = {
        'type': 'choice',
        'instructions': 'Choose the next action from these legal names.',
        'criteria': {name: '' for name in names},
    }
    replay_seed = REPLAY_SEED_BASE + spec['condition_index'] * N_TRAJECTORIES + repetition
    action_rng = np.random.default_rng(replay_seed)
    rewards, expected_rewards = [], []
    elapsed_inference = 0.0
    max_input_tokens = 0
    path = output / f"{spec['condition_id']}_episode_{repetition:02d}.jsonl"
    with path.open('x') as handle:
        for step in range(spec['horizon']):
            state = visible_prompt(prompt_agent)
            started = time.monotonic()
            probability = np.asarray(model.choice_probabilities(state, question, names), dtype=float)
            inference_seconds = time.monotonic() - started
            if (probability.shape != (len(names),) or not np.isfinite(probability).all()
                    or (probability < 0).any() or abs(float(probability.sum()) - 1) > 1e-8):
                raise ValueError('Adapter must return normalized finite legal-choice probabilities.')
            native = getattr(model, 'last_native_choice', None)
            if native is not None and native not in names:
                raise ValueError('Adapter native choice was outside the official action names.')
            if action_rule == 'native':
                if native is None:
                    raise ValueError('Native action requested, but adapter supplied no last_native_choice.')
                name = native
            else:
                name = str(action_rng.choice(names, p=probability))
            action_index = names.index(name)
            _, reward, done, info = env.step(name)
            interaction = info['interaction']
            if info['is_random'] or interaction.mapped_action != action_index:
                raise ValueError('A legal typed choice must not trigger the official invalid-action random fallback.')
            prompt_agent.update(action_index, reward, info)
            if done != (step + 1 == spec['horizon']):
                raise ValueError('The official environment ended at the wrong step.')
            rewards.append(float(reward))
            expected_rewards.append(float(interaction.expected_reward))
            elapsed_inference += inference_seconds
            tokens = getattr(model, 'last_input_tokens', None)
            if tokens is not None:
                max_input_tokens = max(max_input_tokens, int(tokens))
            row = {
                'condition_id': spec['condition_id'], 'repetition_id': repetition,
                'trajectory_seed': seed, 'step': step, 'history': history,
                'action_rule': action_rule, 'replay_action_seed': replay_seed,
                'legal_action_names': names, 'probabilities': probability.tolist(),
                'chosen_action': name, 'chosen_action_index': action_index,
                'native_choice': native, 'reward': float(reward),
                'expected_reward_scoring_only': float(interaction.expected_reward),
                'verbal_feedback': interaction.feedback,
                'input_tokens': int(tokens) if tokens is not None else None,
                'inference_seconds': inference_seconds,
            }
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    if len(rewards) != spec['horizon']:
        raise ValueError('Incomplete trajectory.')
    best_mean = max(spec['arm_expected_rewards_scoring_only'])
    return {
        'condition_id': spec['condition_id'], 'repetition_id': repetition,
        'trajectory_seed': seed, 'steps': len(rewards),
        'cumulative_reward': float(sum(rewards)),
        'cumulative_pseudo_regret': float(sum(best_mean - x for x in expected_rewards)),
        'inference_seconds': elapsed_inference, 'max_input_tokens': max_input_tokens,
        'trajectory_file': path.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-file', type=Path,
                        help='Trusted local create_agent(device) with choice_probabilities.')
    parser.add_argument('--model-label', default='external_frozen_typed_choice')
    parser.add_argument('--history', choices=['raw', 'summary'], default='raw')
    parser.add_argument('--action-rule', choices=['native', 'sample'], default='native')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanity-uniform', action='store_true',
                        help='One complete original 300-step CPU episode, not a scored baseline.')
    args = parser.parse_args()
    if args.sanity_uniform:
        args.adapter_file = ROOT / 'examples/uniform_choice_agent.py'
        args.model_label = 'uniform_interface_sanity_only'
        args.action_rule = 'sample'
        args.device = 'cpu'
    if args.adapter_file is None:
        parser.error('--adapter-file is required outside --sanity-uniform.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep experiment output within this project.')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bench = FullBench(num_threads=1)
    specs = specifications(bench)
    target_indices = [0] if args.sanity_uniform else list(range(16))
    n_episodes = 1 if args.sanity_uniform else N_TRAJECTORIES
    expected_steps = sum(specs[i]['horizon'] for i in target_indices) * n_episodes
    manifest = {
        'status_label': 'interface_sanity_not_baseline' if args.sanity_uniform else SOURCE_LABEL,
        'source': 'https://github.com/allenanie/EVOLvE',
        'suite': 'FullBench MAB verbal environments',
        'public_code_default_episodes_per_condition': bench.n_trials,
        'episodes_per_condition': n_episodes, 'expected_conditions': len(target_indices),
        'expected_episodes': len(target_indices) * n_episodes, 'expected_steps': expected_steps,
        'model_label': args.model_label, 'adapter_file': str(args.adapter_file.resolve()),
        'device': args.device, 'history': args.history,
        'history_source': 'Unmodified LLMMABAgentRH/SH context representation; '
                          'raw retains last 1000 feedback items, equal to full prior history at horizon <=1000; '
                          'summary computes counts and mean rewards from every prior feedback item.',
        'typed_choice_adaptation': 'Original verbal instruction + interaction count + official context + '
                                   'official query are passed as state; legal action names become ordered typed choices. '
                                   'This is an adapted frozen model interface, not original paper LLM decoding.',
        'action_rule': args.action_rule,
        'replay_factor': 'Each policy acts on its own closed-loop observations. Official trajectory seed '
                         'controls reward draws; separately logged action RNG seed is used only when sampling.',
        'trajectory_seed_rule': 'Unmodified get_trajectory_seeds(initial_env_seed,30), not paper trial_id.',
        'horizon_note': 'Public FullBench easy=300/hard=1000; published evaluation protocol differs.',
        'environment_code_modified': False, 'prompt_context_code_modified': False,
        'data_identity': 'Program-generated stochastic bandit; no human target actions.',
        'conditions': [specs[i] for i in target_indices],
    }
    save_json(output / 'manifest.json', manifest)
    status = {'status': 'loading_model', 'completed_conditions': 0, 'completed_episodes': 0,
              'completed_steps': 0, 'expected_steps': expected_steps, 'model_label': args.model_label,
              'history': args.history}
    save_json(output / 'status.json', status)
    rows = []
    current = None
    try:
        model = load_agent(args.adapter_file.resolve(), args.device)
        manifest['model_metadata'] = getattr(model, 'metadata', None)
        save_json(output / 'manifest.json', manifest)
        status['status'] = 'running'
        save_json(output / 'status.json', status)
        for index in target_indices:
            spec = specs[index]
            for repetition in range(n_episodes):
                current = {'condition_id': spec['condition_id'], 'repetition_id': repetition}
                row = run_episode(bench.verbal_envs[index], spec, repetition, model,
                                  args.history, args.action_rule, output)
                rows.append(row)
                status['completed_episodes'] = len(rows)
                status['completed_steps'] += row['steps']
                status['last_completed'] = current
                save_json(output / 'status.json', status)
                print(json.dumps({'status': 'episode_complete', **row}), flush=True)
            status['completed_conditions'] += 1
            save_json(output / 'status.json', status)
        if len(rows) != manifest['expected_episodes'] or sum(r['steps'] for r in rows) != expected_steps:
            raise ValueError('The requested full experiment did not complete.')
        with (output / 'episodes.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        conditions = []
        for index in target_indices:
            selected = [r for r in rows if r['condition_id'] == specs[index]['condition_id']]
            conditions.append({
                'condition_id': specs[index]['condition_id'], 'episodes': len(selected),
                'steps': sum(r['steps'] for r in selected),
                'mean_cumulative_reward': float(np.mean([r['cumulative_reward'] for r in selected])),
                'mean_cumulative_pseudo_regret': float(np.mean([r['cumulative_pseudo_regret'] for r in selected])),
                'max_input_tokens': max(r['max_input_tokens'] for r in selected),
            })
        final_status = 'sanity_complete' if args.sanity_uniform else 'complete'
        summary = {**status, 'status': final_status, 'conditions': conditions,
                   'total_seconds': time.monotonic() - started,
                   'timing_context': 'Shared GPU/CPU; elapsed time is not exclusive throughput.',
                   'metric_note': 'Per-condition cumulative reward and pseudo-regret; do not pool reward scales.',
                   'paper_comparison': 'Official-code FullBench adaptation; not paper-matched horizons/seeds/model decoding.'}
        save_json(output / 'summary.json', summary)
        save_json(output / 'status.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'conditions'}), flush=True)
    except Exception as exc:
        save_json(output / 'status.json', {**status, 'status': 'failed', 'at': current,
                                           'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
