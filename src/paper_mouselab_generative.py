"""Frozen choice agents in Callaway's original Exp1 meta-MDP rollout.

The Julia server executes the published environment and STOP score. A seeded,
pre-sampled reward tree is an explicit equal-distribution replay adaptation;
neither this tree nor the slot identifiers are human behavioral data.
"""
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from src.paper_horizon import load_agent, write_json
from src.paper_mouselab import GRAPH, legal_actions


ROOT = Path(__file__).resolve().parents[1]
N_PARTICIPANTS = 95
N_TRIALS = 25
N_REPEATS = 10
N_EPISODES = N_PARTICIPANTS * N_TRIALS * N_REPEATS
REPLAY_SEED = 20261008
MODEL_COPY = ROOT / 'external/callaway2022_archive/planning/model'
SERVER = ROOT / 'scripts/serve_paper_mouselab_generative.jl'
RULES = (
    'Plan a route from root 0 to a leaf to maximize the expected sum of the 3 visited node values. '
    'Each hidden node independently has value -10, -5, +5, or +10 with equal probability. '
    'Root 0 has known value 0.\n'
    'Tree (parent -> children): 0 -> 1,5,9,13; 1 -> 2; 2 -> 3,4; '
    '5 -> 6; 6 -> 7,8; 9 -> 10; 10 -> 11,12; 13 -> 14; 14 -> 15,16.\n'
    'Before moving, reveal a hidden node only if its parent is known. Revealing deducts no points. '
    'STOP ends revealing; the task automatically chooses the route with the highest expected sum '
    'given the values you have revealed. Unknown node values retain their prior mean of 0.\n'
)


def render_request(history):
    """Render only the model's own observations, using the published tree/prior."""
    keys = legal_actions(history)
    observed = '; '.join(f'{int(n)}={int(v):+d}' for n, v in history) or 'none'
    state = RULES + 'Observed reveals in order: ' + observed + '.'
    question = {
        'type': 'choice', 'instructions': 'Choose your next action in this task.',
        'criteria': {key: ('Stop revealing and execute the best expected route.' if key == 'STOP'
                           else f'Reveal node {key[1:]}.') for key in keys},
    }
    return state, question, keys


def episode_schedule(participants, seed=REPLAY_SEED):
    """Keep the original 95 x 25 x 10 MDP slots, not human hidden rewards."""
    if len(participants) != N_PARTICIPANTS or len(set(participants)) != N_PARTICIPANTS:
        raise ValueError('Expected all 95 original participant MDP slot identifiers.')
    index = 0
    for participant in participants:
        # Original run_simulation uses repeat(mdps, n_repeat): all 25 MDPs
        # for repetition 1, then all 25 for repetition 2, and so on.
        for repeat in range(1, N_REPEATS + 1):
            for trial in range(1, N_TRIALS + 1):
                index += 1
                yield {'participant': participant, 'trial': trial, 'repeat': repeat,
                       'seed': seed + index - 1}


class MouselabServer:
    """One local Julia process, with its diagnostics separate from JSONL replies."""

    def __init__(self, log_path):
        env = os.environ.copy()
        env.update(JULIA_DEPOT_PATH=str(ROOT / '.tools/julia-depot-1.4.1'),
                   JULIA_PROJECT=str(ROOT / '.tools/mouselab-julia141-env'),
                   JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.command = [str(ROOT / '.tools/julia-1.4.1/bin/julia'), '--startup-file=no',
                        '--project=' + env['JULIA_PROJECT'], str(SERVER), str(MODEL_COPY)]
        self.log_path = Path(log_path)
        self.log = self.log_path.open('w')
        self.process = subprocess.Popen(self.command, cwd=MODEL_COPY, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        try:
            self.ready = self.read()
            if self.ready.get('type') != 'ready':
                raise ValueError('Expected the original-environment server ready reply.')
        except Exception:
            self.close(abort=True)
            raise

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Mouselab Julia server stopped; see {self.log_path}.')
        return json.loads(line)

    def request(self, command, **fields):
        self.process.stdin.write(json.dumps({'op': command, **fields}, allow_nan=False) + '\n')
        self.process.stdin.flush()
        reply = self.read()
        if reply.get('type') == 'error':
            raise RuntimeError(f'Mouselab Julia error: {reply}; see {self.log_path}.')
        return reply

    def close(self, abort=False):
        if self.process.poll() is None:
            if abort:
                self.process.terminate()  # Only the child owned by this runner.
            else:
                self.process.stdin.write(json.dumps({'op': 'quit'}) + '\n')
                self.process.stdin.flush()
        exit_code = self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
        if not abort and exit_code:
            raise RuntimeError(f'Mouselab Julia exited with code {exit_code}; see {self.log_path}.')


def expected_route_value(history):
    """Independent direct check of the official belief-based STOP score."""
    known = {0: 0, **{int(n): int(v) for n, v in history}}

    def value(node):
        children = GRAPH[node]
        return known.get(node, 0) + (max(value(child) for child in children) if children else 0)

    return value(0)


def check_terminal(reply, history):
    """Catch index/feedback/score mistakes before accepting an episode."""
    if not reply['done'] or reply['reveals'] != len(history):
        raise ValueError('Original rollout must end in STOP with all own reveals accounted for.')
    if float(reply['terminal_expected_reward']) != expected_route_value(history):
        raise ValueError('STOP score differs from the visible-belief best expected route.')
    truth = reply['ground_truth']
    if len(truth) != 17 or truth[0] != 0 or any(v not in [-10, -5, 5, 10] for v in truth[1:]):
        raise ValueError('Generated hidden tree must retain the original independent discrete prior.')
    if any(truth[int(node)] != value for node, value in history):
        raise ValueError('Observed feedback differs from the generated episode tree.')


def agent_metadata(agent):
    metadata = dict(getattr(agent, 'metadata', {}))
    # Laya's existing adapter exposes the SDK through agent.agent.
    if hasattr(agent, 'agent') and hasattr(agent.agent, 'cfg'):
        sdk = agent.agent
        metadata.update(model_directory=str(ROOT / 'models/laya-english'),
                        actual_parameter_dtypes=sorted({str(p.dtype) for p in sdk.model.parameters()}),
                        max_len=sdk.cfg['max_len'], head_max_len=sdk.cfg['head_max_len'],
                        forward_dtype=str(sdk.dtype), temperature_config=sdk.cfg.get('temperature'),
                        temperature_by_options=sdk.cfg.get('temperature_by_options'),
                        training='Frozen public weights; no task fine-tuning or temperature search.',
                        probabilities='Native SDK four-decimal values, explicitly normalized by existing adapter.',
                        native_choice='Native SDK choice, retained before probability-rounding tie handling.')
    return metadata


def run_mouselab_generative(output, adapter_file, device='cpu', action_rule='native', seed=REPLAY_SEED):
    """Run the complete original fitted-policy simulation scale for one new model."""
    output, adapter_file = Path(output), Path(adapter_file)
    if action_rule not in {'native', 'sample'}:
        raise ValueError('action_rule must be native or sample.')
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    config = {
        'status': 'loading', 'task': 'mouselab-generative', 'mode': 'generative',
        'paper': 'https://doi.org/10.1038/s41562-022-01332-8',
        'code_source': 'external/callaway2022_archive/planning/model',
        'original_environment_copy': str(MODEL_COPY), 'server_source': str(SERVER),
        'data_kind': 'new_program_generated_reward_trees_not_human_trajectories',
        'participants_as_schedule_slots': N_PARTICIPANTS, 'trials_per_slot': N_TRIALS,
        'repeats_per_trial': N_REPEATS, 'expected_episodes': N_EPISODES,
        'adapter_file': str(adapter_file.resolve()), 'device': device, 'action_rule': action_rule,
        'replay_seed': seed, 'episode_seed_rule': 'replay_seed + zero-based episode index',
        'replay_adaptation': 'Pre-sample the independent node rewards through the original Julia prior '
            'and use original rollout(policy, hidden_state). This is equal in distribution to the '
            'original on-demand sampling; the original simulate.jl does not set a replay seed.',
        'environment_cost': 0.0,
        'reward_semantics': 'Original best-path expected value at STOP, not realized route payoff.',
        'stopping_semantics': 'Original meta-MDP automatically executes the best expected route; '
            'a human route-selection action is not evaluated here.',
        'cost_semantics': 'Report reveal count separately. No imposed lambda, actual 3-second sleep, '
            'or conversion of human waiting seconds into points.',
        'interface_adaptation': 'New explicit text and typed choices for the published meta-MDP; '
            'not an original human prompt or original decoder reproduction.',
        'visible_information': 'Tree, public independent prior, legal actions and complete own reveal history. '
            'No generated hidden rewards, human actions, participant identifiers or replay seeds sent to model.',
        'history': 'Complete current-episode own reveal history; reset between episodes.',
        'native_action_definition': 'Use last_native_choice when supplied; otherwise probability argmax '
            'with ties taking the first ordered key. Log the actual action source.',
        'sample_action_definition': 'External sampling of submitted probabilities using a separate per-episode '
            'NumPy RNG; this is a distinct execution condition.',
        'timing_context': 'Shared CPU/GPU; these measured times are not exclusive-device performance.',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid(),
    }
    write_json(output / 'config.json', config)
    write_json(output / 'status.json', {'status': 'loading', 'expected_episodes': N_EPISODES,
                                      'supervisor_pid': os.getpid()})
    server = None
    completed = 0
    decisions = 0
    inference_seconds = 0.0
    reward_sum = 0.0
    reveal_sum = 0
    action_sources = Counter()
    try:
        server_started = time.monotonic()
        server = MouselabServer(output / 'environment.log')
        config['server_start_seconds'] = time.monotonic() - server_started
        config['server_metadata'] = server.ready
        config['server_command'] = server.command
        participants = server.ready['participants']
        if (server.ready['trials_per_participant'], server.ready['repeats']) != (N_TRIALS, N_REPEATS):
            raise ValueError('Original full MDP schedule changed.')
        config['participant_slot_order'] = participants
        loading_started = time.monotonic()
        agent = load_agent(adapter_file, device)
        config.update(status='running', load_seconds=time.monotonic() - loading_started,
                      model_metadata=agent_metadata(agent))
        write_json(output / 'config.json', config)
        with ExitStack() as files:
            requests = files.enter_context((output / 'requests.jsonl').open('w'))
            probabilities = files.enter_context((output / 'probabilities.jsonl').open('w'))
            targets = files.enter_context((output / 'targets.jsonl').open('w'))
            trials = files.enter_context((output / 'trials.csv').open('w', newline=''))
            writer = csv.DictWriter(trials, fieldnames=['episode', 'participant_slot', 'trial_slot', 'repeat',
                                                      'seed', 'terminal_expected_reward', 'reveals',
                                                      'decisions', 'inference_seconds'])
            writer.writeheader()
            for episode, slot in enumerate(episode_schedule(participants, seed), 1):
                reply = server.request('reset', **slot)
                history = []
                episode_decisions = 0
                episode_seconds = 0.0
                action_rng = np.random.RandomState(slot['seed'] + N_EPISODES)
                while not reply['done']:
                    if reply['history'] != history:
                        raise ValueError('Original environment reply does not match own observation history.')
                    state, question, keys = render_request(history)
                    if reply['legal_actions'] != keys:
                        raise ValueError('Text actions differ from the original Julia legal-action frontier.')
                    decision_id = f'{episode}/{episode_decisions + 1}'
                    requests.write(json.dumps({'id': decision_id, 'state': state, 'question': question,
                                               'output_keys': keys}, ensure_ascii=False, allow_nan=False) + '\n')
                    begin = time.monotonic()
                    p = np.asarray(agent.choice_probabilities(state, question, keys.copy()), dtype=float)
                    seconds = time.monotonic() - begin
                    if (p.shape != (len(keys),) or not np.isfinite(p).all() or (p < 0).any()
                            or (p > 1).any() or abs(p.sum() - 1) > 1e-8):
                        raise ValueError('Return a normalized finite probability for each legal action.')
                    native_choice = getattr(agent, 'last_native_choice', None)
                    if native_choice is not None and native_choice not in keys:
                        raise ValueError('Native choice must be a current legal action.')
                    if action_rule == 'sample':
                        choice, source = str(action_rng.choice(keys, p=p)), 'external_probability_sampling'
                    elif native_choice is not None:
                        choice, source = native_choice, 'native_choice'
                    else:
                        choice, source = keys[int(p.argmax())], 'submitted_probability_argmax'
                    row = {'id': decision_id, 'episode': episode, 'observed_history': history.copy(),
                           'probabilities': dict(zip(keys, p.tolist())), 'choice': choice,
                           'native_choice': native_choice, 'action_source': source,
                           'inference_seconds': seconds}
                    for name in ['last_input_tokens', 'last_prompt', 'last_label_mapping',
                                 'last_allowed_token_mass', 'last_raw_probabilities']:
                        if hasattr(agent, name):
                            row[name.removeprefix('last_')] = getattr(agent, name)
                    probabilities.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    next_reply = server.request('step', action=choice)
                    episode_decisions += 1
                    decisions += 1
                    episode_seconds += seconds
                    inference_seconds += seconds
                    action_sources[source] += 1
                    if choice == 'STOP':
                        if not next_reply['done']:
                            raise ValueError('STOP must terminate the original rollout.')
                    else:
                        if next_reply['done'] or next_reply['history'][:-1] != history:
                            raise ValueError('Reveal must add exactly one visible observation without terminating.')
                        node, value = next_reply['history'][-1]
                        if node != int(choice[1:]) or value not in [-10, -5, 5, 10]:
                            raise ValueError('Original reveal feedback must match the chosen node and prior.')
                        history = next_reply['history']
                    reply = next_reply
                check_terminal(reply, history)
                if episode_decisions != len(history) + 1:
                    raise ValueError('Each episode must contain all own reveals followed by one STOP.')
                targets.write(json.dumps({'id': str(episode), 'metadata': slot,
                                          'ground_truth': reply['ground_truth'],
                                          'terminal_belief': reply['terminal_belief'],
                                          'official_best_path': reply['official_best_path'],
                                          'terminal_expected_reward': reply['terminal_expected_reward']},
                                         ensure_ascii=False, allow_nan=False) + '\n')
                writer.writerow({'episode': episode, 'participant_slot': slot['participant'],
                                 'trial_slot': slot['trial'], 'repeat': slot['repeat'], 'seed': slot['seed'],
                                 'terminal_expected_reward': reply['terminal_expected_reward'],
                                 'reveals': len(history), 'decisions': episode_decisions,
                                 'inference_seconds': episode_seconds})
                completed += 1
                reward_sum += reply['terminal_expected_reward']
                reveal_sum += len(history)
                if completed % 50 == 0 or completed == N_EPISODES:
                    for file in [requests, probabilities, targets, trials]:
                        file.flush()
                    progress = {'status': 'running', 'completed_episodes': completed,
                                'expected_episodes': N_EPISODES, 'completed_decisions': decisions,
                                'mean_terminal_expected_reward': reward_sum / completed,
                                'mean_reveals': reveal_sum / completed,
                                'elapsed_seconds': time.monotonic() - started,
                                'supervisor_pid': os.getpid(), 'environment_pid': server.process.pid}
                    write_json(output / 'progress.json', progress)
                    write_json(output / 'status.json', progress)
                    print(json.dumps(progress, allow_nan=False), flush=True)
        if completed != N_EPISODES:
            raise ValueError('The complete 23750-episode task was not executed.')
        server.close()
        server = None
        summary = {
            'status': 'complete', 'task': 'mouselab-generative', 'simulations': completed,
            'participants_as_schedule_slots': N_PARTICIPANTS, 'trials_per_slot': N_TRIALS,
            'repeats_per_trial': N_REPEATS, 'decisions': decisions,
            'mean_terminal_expected_reward': reward_sum / completed,
            'mean_reveals': reveal_sum / completed, 'environment_cost': 0.0,
            'action_rule': action_rule, 'action_source_counts': dict(action_sources),
            'adapter_file': config['adapter_file'], 'model_metadata': config['model_metadata'],
            'total_seconds': time.monotonic() - started, 'load_seconds': config['load_seconds'],
            'server_start_seconds': config['server_start_seconds'], 'inference_seconds': inference_seconds,
            'maximum_input_tokens': getattr(agent, 'max_input_tokens', None),
            'timing_context': config['timing_context'], 'reward_semantics': config['reward_semantics'],
            'replay_adaptation': config['replay_adaptation'], 'replay_seed': seed,
            'data_kind': config['data_kind'], 'interface_adaptation': config['interface_adaptation'],
            'checks': 'Every original schedule slot completed; current legal frontier, own feedback, '
                'hidden-tree reward consistency, one final STOP and belief-based expected score checked.',
            'comparison_limits': 'Original fitted policies have individual fitted costs; compare gross '
                'expected reward and reveals separately. No matched-lambda dominance or architecture '
                'causality follows. Existing unseeded official results are not paired with these trees.',
        }
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', summary)
        write_json(output / 'progress.json', summary)
        config['status'] = 'complete'
        write_json(output / 'config.json', config)
        print(json.dumps(summary, allow_nan=False), flush=True)
        return summary
    except Exception as exc:
        failure = {'status': 'failed', 'task': 'mouselab-generative',
                   'completed_episodes': completed, 'expected_episodes': N_EPISODES,
                   'completed_decisions': decisions, 'error': str(exc),
                   'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started}
        write_json(output / 'status.json', failure)
        write_json(output / 'failure.json', failure)
        if server is not None:
            server.close(abort=True)
        raise
