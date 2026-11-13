"""Synthetic test fixture, not a benchmark task. A two-arm environment with its own run()."""

TASK = {
    'id': 'toy_environment',
    'label': 'Toy environment',
    'paradigm': 'Synthetic two-arm environment, test fixture only',
    'source': 'None. The rewards below are invented for the offline tests.',
    'license': 'MIT, written for this repository',
    'split': 'Five synthetic steps',
    'obtain': 'Bundled in this file.',
    'adaptations': 'None, the fixture is defined in text.',
    'inputs': ['tests/fixtures/extensions/toy_environment/task.py'],
    'n_inputs': 5,
    'n_scored': 5,
    'autonomous': True,
    'interface': 'choice_probabilities',
    'direction': 'higher',
    'protocol': 'toy_two_arm',
    'mode': 'autonomous_generated_environment',
    'metric': 'reward',
    'metric_variant': 'mean_reward',
    'unit': 'task_specific_reward_units',
    'aggregation': 'response_weighted_mean',
    'target': 'autonomous_environment_reward',
    'reference': {'value': 1.0, 'description': 'Always choosing arm B',
                  'provenance': 'Analytic solution'},
}
REWARD = {'A': 0.0, 'B': 1.0}


def run(agent, out, limit, device, action_rule):
    steps = 5 if limit is None else min(limit, 5)
    total = 0.0
    for t in range(steps):
        probabilities = agent.choice_probabilities(f'Step {t}.', 'Choose A or B.', ['A', 'B'])
        choice = getattr(agent, 'last_native_choice', None) or ('A' if probabilities[0] >= probabilities[1] else 'B')
        total += REWARD[choice]
    return {'inputs': steps, 'scored': steps, 'value': total / steps, 'n': steps,
            'summary': {'steps': steps, 'total_reward': total}}
