"""The 27 evaluated modes of the 24 first-paper tasks.

Each mode names the original runner family that the 24-task CPU validation
driver (scripts/validate_cpu_reuse_24.py) dispatched to, the adapter methods
that runner calls, the inputs it reads relative to the data root, the
full-sample counts, and the comparison context of its Table 3 primary target
as written in results/aggregate/results.json.
"""
from dataclasses import dataclass

PSYCH101_TEST = 'data/psych101/prompts_testing_t1.jsonl'
COGBENCH = 'external/CogBench/Experiments'
CENTAUR = 'external/paper_matched_centaur_evaluation'
COGGYM_CODE = 'external/coggym_social_evaluation'
COGGYM_RAW = 'literature/task_catalog_20261017/raw'
FOUR = 'literature/four_task_resolution_20261018'
MOUSELAB = 'data/mouselab_callaway2022_official'
CALLAWAY = 'external/callaway2022_archive/planning'
WCST_INPUTS = (f'{CENTAUR}/wcst/Data/json_wcst_dataset.npy',
               f'{CENTAUR}/wcst/generative_predictive_llms_eval/predictive_centaur_70B.py',
               f'{CENTAUR}/wcst/generative_predictive_llms_eval/generative_centaur_70B.py')
JULIA = ('.tools/julia-1.4.1/bin/julia', '.tools/mouselab-julia141-env/Project.toml',
         '.tools/mouselab-julia141-env/Manifest.toml', '.tools/julia-depot-1.4.1')

P = 'heldout_human_history_prediction'
A = 'aggregate_human_distribution_comparison'
HUMAN = 'observed_human_response'
LABEL = 'original_objective_answer_or_optimal_action_label'
SOCIAL = 'social-public-full-materials-typed-v1'
TYPED = 'original_full_prefix_typed'
FOUR_PROTOCOL = 'full-original-prefix-integer-and-self-assessment-v1'
CHOICE = ('choice_probabilities',)
NATIVE = ('choice_probabilities', 'last_native_choice')
EVENT = ('posterior_probability',)
# Metrics read from action probabilities. A generated-text adapter supplies
# none, so these are left undefined for it instead of scored on a placeholder.
PROBABILITY_METRICS = {'human_nll', 'aggregate_distribution_nll'}


def coggym(*studies):
    files = [f'{COGGYM_RAW}/coggym_repo__evaluation__trial_selection_map.json']
    for study in studies:
        files += [f'{COGGYM_RAW}/coggym_repo__EML__{study}__{name}' for name in (
            'config.json', 'instruction.jsonl', 'trial.jsonl', 'human_data_mean.json',
            'human_data_ind.json')]
    return (COGGYM_CODE, *files)


def context(protocol, mode, metric, variant, unit, aggregation, target, scope='all'):
    return {'protocol': protocol, 'mode': mode, 'metric': metric, 'metric_variant': variant,
            'scope': scope, 'unit': unit, 'aggregation': aggregation, 'target': target}


@dataclass(frozen=True)
class Mode:
    task: str
    mode: str
    family: str
    label: str
    interface: tuple
    inputs: tuple
    inputs_expected: int  # full input count checked by the CPU validation
    scored_expected: int  # original scored count checked by the CPU validation
    primary: dict         # comparison context of the Table 3 primary target
    note: str = ''
    track: str = 'core'   # 'core' for the 27 paper modes, 'extension' for extensions/<task>/task.py
    direction: str = ''   # better direction of an extension target; core directions are in table2.json

    @property
    def key(self):
        return self.task if self.mode == 'primary_protocol' else f'{self.task}:{self.mode}'


def typed(task, label, n, primary, inputs=(PSYCH101_TEST,), interface=CHOICE):
    return Mode(task, 'primary_protocol', 'typed', label, interface, inputs, n, n, primary)


MODES = [
    Mode('bayesian_urn_inference', 'primary_protocol', 'posterior', 'Bayesian urn', EVENT,
         (f'{COGBENCH}/ProbabilisticReasoning',), 400, 400,
         context('cogbench_generated_posterior_200x2', 'generated_explicit_posterior', 'mae',
                 'posterior_mae', 'probability_0_to_1', 'response_weighted_mean',
                 'analytic_Bayes_posterior')),
    typed('multiple_cue_judgment', 'Multiple-cue judgment', 6136,
          context('compact_full_history', P, 'human_nll', 'typed_choice_mean_nll',
                  'nats_per_response', 'response_weighted_mean', HUMAN)),
    typed('multi_attribute_choice', 'Multi-attribute choice', 576,
          context(TYPED, P, 'human_nll', 'typed_choice_mean_nll', 'nats_per_response',
                  'response_weighted_mean', HUMAN)),
    Mode('probability_coherence_judgment', 'primary_protocol', 'four', 'Probability coherence',
         CHOICE, (PSYCH101_TEST,), 2460, 2460,
         context(FOUR_PROTOCOL, P, 'mae', 'mean_numeric_absolute_error',
                 'probability_judgment_percentage_points_0_to_100', 'response_weighted_mean',
                 'original_human_numeric_response')),
    typed('shj_category_learning', 'SHJ categories', 3664,
          context(TYPED, P, 'accuracy', 'category_accuracy_under_human_history', 'fraction',
                  'response_weighted_mean', LABEL)),
    typed('medin_category_learning', 'Medin categories', 6972,
          context(TYPED, P, 'accuracy', 'objective_label_accuracy_under_human_history',
                  'fraction', 'response_weighted_mean', LABEL)),
    typed('weather_prediction', 'Weather prediction', 400,
          context(TYPED, P, 'accuracy', 'objective_label_accuracy_under_human_history',
                  'fraction', 'response_weighted_mean', LABEL)),
    typed('rlwm', 'RLWM', 52911,
          context(TYPED, 'original_human_history_replay_no_official_heldout_split', 'accuracy',
                  'objective_label_accuracy_under_human_history', 'fraction',
                  'response_weighted_mean', LABEL),
          inputs=('data/rlwm/CollinsFrank_2012_EJoN.csv',)),
    typed('recent_probes', 'Recent probes', 3774,
          context(TYPED, P, 'accuracy', 'objective_label_accuracy_under_human_history',
                  'fraction', 'response_weighted_mean', LABEL)),
    Mode('wcst', 'human_predictive', 'wcst', 'WCST, prediction', NATIVE,
         WCST_INPUTS, 12754, 12754,
         context('wcst_predictive_75_test_participants', 'predictive', 'human_nll',
                 'participant_mean', 'nats_per_response', 'participant_mean', HUMAN,
                 scope='predictive')),
    Mode('wcst', 'autonomous_generative', 'wcst', 'WCST, autonomous', NATIVE,
         WCST_INPUTS, 12754, 12754,
         context('wcst_generative_75_test_participants', 'generative', 'accuracy',
                 'participant_mean', 'fraction', 'participant_mean', 'original_rule_correct_card',
                 scope='generative')),
    Mode('nonstationary_bandit', 'primary_protocol', 'restless', 'Restless bandit',
         ('choice_probabilities', 'confidence_probability'), (f'{COGBENCH}/RestlessBandit',),
         1600, 1600,
         context('cogbench_restless_action_and_confidence_10x80',
                 'autonomous_generated_environment', 'accuracy', 'accuracy_post_step_original',
                 'fraction', 'mean_of_10_equal_length_80_trial_run_scores', LABEL)),
    Mode('instrumental_learning', 'primary_protocol', 'expansion', 'Instrumental learning',
         CHOICE, (f'{COGBENCH}/InstrumentalLearning',), 960, 960,
         context('original_environment', 'autonomous_generated_environment', 'accuracy',
                 'original_latent_mean_optimal_action_rate', 'fraction',
                 'response_weighted_mean', 'maximal_original_latent_arm_mean')),
    Mode('horizon', 'human_predictive', 'prediction', 'Horizon, prediction', CHOICE,
         (f'{CENTAUR}/horizon/data/in/test_data.csv',
          f'{CENTAUR}/horizon/predictive_horizon_centaur.py',
          f'{COGBENCH}/HorizonTask/envs/bandits.py'), 6720, 6720,
         context('namazova_human_predictive_6x1120', P, 'human_nll', 'legal_nll_eps1e8',
                 'nats_per_response', 'response_weighted_mean', HUMAN)),
    Mode('horizon', 'autonomous_generative', 'horizon_generative', 'Horizon, autonomous', CHOICE,
         (f'{CENTAUR}/horizon/data/in/timeline_structure.csv', f'{CENTAUR}/horizon/horizon_centaur_one.py',
          f'{COGBENCH}/HorizonTask/envs/bandits.py'), 34720, 34720,
         context('namazova_generative_31_timelines_{rule}', 'autonomous_generative', 'accuracy',
                 'optimal_action_rate', 'fraction', 'response_weighted_mean',
                 'latent_mean_optimal_arm', scope='free_choices')),
    Mode('stationary_multiarmed_bandit', 'primary_protocol', 'bandit', 'Stationary bandits',
         CHOICE, ('external/EVOLvE',), 16, 312000,
         context('official_code_fullbench_summary_history_16_conditions', 'autonomous',
                 'accuracy', 'accuracy', 'proportion', 'response_weighted_mean',
                 'maximal_original_latent_arm_mean', scope='all_16_original_conditions'),
         note='The native action rule needs last_native_choice; --action-rule sample does not.'),
    Mode('hybrid_information_sampling', 'primary_protocol', 'hybrid', 'Hybrid sampling', CHOICE,
         ('external/HybridModellingProject/Data/fit/data',
          'data/hybrid_paper_matched/fold_assignments.csv'), 163135, 163135,
         context('original_20_subjects_8fold_frozen_prediction', P, 'human_nll',
                 'pooled_action_nll', 'nats_per_response', 'response_weighted_mean', HUMAN)),
    Mode('mouselab_tree', 'human_predictive', 'prediction', 'Mouselab, prediction', CHOICE,
         (f'{MOUSELAB}/trials.json', f'{MOUSELAB}/folds.json'), 20363, 20363,
         context('original_human_predictive_95x25', P, 'human_nll', 'mean_nll',
                 'nats_per_response', 'response_weighted_mean', HUMAN)),
    Mode('mouselab_tree', 'autonomous_generative', 'mouselab_generative', 'Mouselab, autonomous',
         CHOICE, (f'{MOUSELAB}/trials.json', f'{CALLAWAY}/model', f'{CALLAWAY}/data/exp1/trials.json',
                  *JULIA), 23750, 23750,
         context('original_metamdp_generative_23750_{rule}', 'autonomous_generative', 'reward',
                 'mean_terminal_expected_reward', 'task_specific_reward_units', 'episode_mean',
                 'best_path_expected_value_at_STOP', scope='native_autonomous_full'),
         note='Needs the Julia 1.4.1 runtime and its package depot (see docs/DATA.md).'),
    Mode('bart', 'primary_protocol', 'expansion', 'BART', CHOICE, (f'{COGBENCH}/BART',), 900, 900,
         context('original_environment', 'autonomous_generated_environment', 'reward',
                 'mean_reward', 'task_specific_reward_units', 'response_weighted_mean',
                 'autonomous_environment_reward'),
         note='The accepted results use 20 runs (900 balloons). The 10-26 CPU validation ran '
              'the 10-run code default (450).'),
    Mode('temporal_discounting', 'primary_protocol', 'expansion', 'Temporal discounting', CHOICE,
         (f'{COGBENCH}/TemporalDiscounting',), 1, 1,
         context('original_environment', 'autonomous_generated_environment', 'preference',
                 'delayed_reward_choice_rate', 'fraction', 'response_weighted_mean',
                 'later_receipt_among_reward_choice_questions',
                 scope='reward_choice_questions'),
         note='Preference, no better direction. Table 3 definition of 2026-10-29: payment-only '
              'questions are excluded.'),
    Mode('two_step', 'primary_protocol', 'twostep', 'Two-step', CHOICE, (f'{COGBENCH}/TwoStepTask',),
         4000, 4000,
         context('original_CogBench_v1_environment_with_new_typed_choice_adapter',
                 'autonomous_generated_environment', 'reward', 'mean_reward',
                 'task_specific_reward_units', 'response_weighted_mean',
                 'autonomous_environment_reward')),
    typed('false_belief_story_qa', 'False-belief QA', 5994,
          context(SOCIAL, 'public_scenario_qa', 'accuracy', 'original_answer_exact_match_mean',
                  'fraction', 'response_weighted_mean', LABEL),
          inputs=(COGGYM_CODE, 'data/tomi_original_2019/tomi_balanced_story_types/fb_all_test.txt',
                  'data/tomi_original_2019/tomi_balanced_story_types/fb_all_test.trace')),
    typed('politeness_pragmatics', 'Politeness', 128,
          context(SOCIAL, A, 'aggregate_distribution_nll', 'human_response_weighted_mean_nll',
                  'nats_per_response', 'response_weighted_mean',
                  'public_aggregate_human_distribution_not_individual_holdout_response'),
          inputs=coggym('Yoon2020Polite__exp1', 'Yoon2020Polite__exp2')),
    typed('pragmatic_language_interpretation', 'Pragmatic interpretation', 170,
          context(SOCIAL, A, 'aggregate_distribution_nll', 'human_response_weighted_mean_nll',
                  'nats_per_response', 'response_weighted_mean',
                  'public_aggregate_human_distribution_not_individual_holdout_response'),
          inputs=coggym('Hu2023Fine__exp1')),
    typed('nonliteral_number_interpretation', 'Nonliteral number', 330,
          context(SOCIAL, A, 'mae', 'absolute_error_to_human_mean_mean',
                  'reported_percentage_points_0_to_100', 'response_weighted_mean',
                  'public_human_mean_slider_judgment'),
          inputs=coggym('Tsvilodub2025Nonliteral__exp1', 'Tsvilodub2025Nonliteral__exp3b'),
          interface=EVENT),
    Mode('performance_self_assessment', 'primary_protocol', 'four', 'Self-assessment', CHOICE,
         (PSYCH101_TEST, *(f'{FOUR}/{name}' for name in (
             'Grammar_SA_MTurk_-_attchecks.qsf', 'LSAT_SA_MTurk_-_attchecks.qsf',
             'openloop__jansen2021dunningkruger__exp1.csv',
             'generalization__jansen2021logic__exp1.csv',
             'generalization__jansen2021logic__prompts.jsonl'))), 108248, 108248,
         context(FOUR_PROTOCOL, P, 'human_nll', 'mean_typed_nll', 'nats_per_response',
                 'response_weighted_mean', HUMAN)),
]
BY_KEY = {m.key: m for m in MODES}
TASKS = list(dict.fromkeys(m.task for m in MODES))
assert len(MODES) == 27 and len(TASKS) == 24 and len(BY_KEY) == 27


def select(names):
    """Resolve 'all', task IDs (every mode of the task) and task:mode keys."""
    if names in (['all'], 'all'):
        return list(MODES)
    chosen = []
    for name in names:
        hits = [m for m in MODES if name in (m.key, m.task)]
        if not hits:
            raise SystemExit(f'Unknown task or mode {name!r}. Run `python -m s1cogbench list`.')
        chosen += [m for m in hits if m not in chosen]
    return chosen
