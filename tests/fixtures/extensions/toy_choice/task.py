"""Synthetic test fixture, not a benchmark task. Four made-up human choices over three keys."""

TASK = {
    'id': 'toy_choice',
    'label': 'Toy choice',
    'paradigm': 'Synthetic three-option choice, test fixture only',
    'source': 'None. The trials below are invented for the offline tests.',
    'license': 'MIT, written for this repository',
    'split': 'All four synthetic trials',
    'obtain': 'Bundled in this file.',
    'adaptations': 'None, the fixture is defined in text.',
    'inputs': ['tests/fixtures/extensions/toy_choice/task.py'],
    'n_inputs': 4,
    'n_scored': 4,
    'protocol': 'toy_full_history_v1',
    'mode': 'heldout_human_history_prediction',
    'metric': 'human_nll',
    'metric_variant': 'typed_choice_mean_nll',
    'unit': 'nats_per_response',
    'aggregation': 'response_weighted_mean',
    'target': 'observed_human_response',
    'reference': {'value': 1.0986, 'description': 'Uniform choice over three keys, ln 3',
                  'provenance': 'Analytic expectation'},
}


def load_trials(root):
    history = ''
    trials = []
    for i, observed in enumerate('ABCA'):
        trials.append({'id': f'p1-t{i}', 'state': f'Earlier choices: {history or "none"}.',
                       'question': 'Choose A, B or C.', 'output_keys': ['A', 'B', 'C'],
                       'observed': observed})
        history += observed
    return trials
