"""Template for an extension task. Copy this directory to extensions/<task_id>/ and replace every value.

Directories whose names start with an underscore are not loaded, so this file never
enters `python -m s1cogbench list`. docs/ADD_TASK.md explains each field and the
review criteria. Keep imports light and read data only inside load_trials() or run(),
because the command line imports this file to list tasks.
"""

TASK = {
    'id': 'my_task',                     # equals the directory name; lowercase letters, digits, underscores
    'label': 'My task',                  # short name shown on the leaderboard
    'paradigm': 'Original experiment and what it manipulates, in one or two sentences.',
    'source': 'Author et al. (Year). Title. Venue. Data DOI or pinned URL.',
    'license': 'Data license, and whether any file is redistributed in this directory.',
    'split': 'Official split used, for example the 40 test participants of the original release.',
    'obtain': 'How a user obtains each input path below at a pinned revision.',
    'adaptations': 'Every change from the original, such as text display, supplied histories or constrained responses.',
    'inputs': ['extensions/my_task/inputs/trials.csv'],  # checked by `doctor`; not under a directory named data/
    'n_inputs': 1000,                    # full source units read, checked on every full run
    'n_scored': 1000,                    # full scored responses, checked on every full run
    'protocol': 'my_task_original_v1',   # change it whenever the protocol changes
    'mode': 'heldout_human_history_prediction',
    'metric': 'human_nll',               # human_nll, accuracy or probability_mae; anything else needs run()
    'metric_variant': 'typed_choice_mean_nll',
    'unit': 'nats_per_response',
    'aggregation': 'response_weighted_mean',
    'target': 'observed_human_response',
    'reference': {'value': 0.0,          # at least one reference score with its provenance
                  'description': 'What the reference is, for example a model fitted to training participants',
                  'provenance': 'Official code, disclosed reimplementation, analytic solution or answer key'},
}


def load_trials(root):
    """Return every scored trial in source order. `root` is the repository root.

    Each trial is a dictionary with a unique source `id`, the visible `state` text,
    the `question` text, the ordered legal `output_keys`, and the target field of the
    metric: `observed` (human_nll, the participant's key), `correct` (accuracy) or
    `target` (probability_mae, a probability in [0, 1]). Only state, question and
    output_keys reach the model.
    """
    raise NotImplementedError('Read the inputs listed in TASK and build the trials.')


# A task whose score is not one of the built-in metrics, or that runs an environment
# on the model's own actions, defines run() instead of load_trials(). It then also sets
# TASK['interface'] (the adapter method it calls), TASK['direction'] (lower, higher or
# none) and, for environments, TASK['autonomous'] = True so that --action-rule applies.
#
# def run(agent, out, limit, device, action_rule):
#     """Run the full protocol, or at most `limit` model calls when limit is not None.
#     Write per-response records under `out`. Return
#     {'inputs': ..., 'scored': ..., 'value': primary score, 'n': its denominator,
#      'summary': dict of further fields}."""
