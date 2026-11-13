"""Synthetic test fixture, not a benchmark task. Three made-up probability targets."""

TASK = {
    'id': 'toy_report',
    'label': 'Toy report',
    'paradigm': 'Synthetic probability report, test fixture only',
    'source': 'None. The targets below are invented for the offline tests.',
    'license': 'MIT, written for this repository',
    'split': 'All three synthetic items',
    'obtain': 'Bundled in this file.',
    'adaptations': 'None, the fixture is defined in text.',
    'inputs': ['tests/fixtures/extensions/toy_report/task.py'],
    'n_inputs': 3,
    'n_scored': 3,
    'protocol': 'toy_reports_v1',
    'mode': 'generated_explicit_posterior',
    'metric': 'probability_mae',
    'metric_variant': 'posterior_mae',
    'unit': 'probability_0_to_1',
    'aggregation': 'response_weighted_mean',
    'target': 'analytic_probability',
    'reference': {'value': 0.0, 'description': 'The analytic probabilities themselves',
                  'provenance': 'Analytic solution'},
}


def load_trials(root):
    return [{'id': f'item-{i}', 'state': f'Item {i}.', 'question': 'Report the probability.',
             'target': target} for i, target in enumerate((0.1, 0.5, 0.8))]
