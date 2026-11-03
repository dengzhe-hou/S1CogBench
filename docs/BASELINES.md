# Task-specific references

The five open models and a cognitive reference have different roles. A reference instantiates a task-compatible algorithm or fitted behavioral model. The scorer evaluates its outputs; it is not itself a competing decision model.

The [reference results](../results/aggregate/reference_results.json) and [provenance](../results/aggregate/baseline_provenance.json) retain fitting exposure, target, n, aggregation and reported precision. Every Reference and Uniform cell of Table 3 is a row there, and [table2.json](../results/aggregate/table2.json) names which. Official code, disclosed reimplementations and archived-parameter replays stay separate. A human-action NLL is not an autonomous-return comparator.

Horizon RW3, WCST SL, Hybrid independent ANN, multiple-cue updating and Mouselab MetaGreedy retain the original participant/fold rules. RLWM parameter replays do not establish a matched held-out neural ranking. The Bandit download retains eight Gaussian Thompson conditions with the declared NIG prior and all original seeds.

The different six-parameter Horizon RW is outside the fixed phase. A historical ledger blank does not automatically create a new experiment.

## Uniform column

Uniform integrates over legal actions or reports with source weights and tie credit. These are analytical expectations under the original scoring rules, not additional sampled model runs. A run of `examples/uniform_reference_agent.py` through the CLI reproduces 18 of them ([REPRODUCTION.md](REPRODUCTION.md)). The other nine are autonomous modes, where one rollout is only a sample of the expectation. The values are in [reference_results.json](../results/aggregate/reference_results.json) under `reference_id` `uniform_choice`, printed here at Table 3's precision.

| Mode | Uniform | Definition | Reproduced by a run |
| --- | ---: | --- | --- |
| Bayesian urn | 0.349 | Continuous uniform report on [0, 1]. The expected absolute error for posterior p is (p^2 + (1 - p)^2) / 2, averaged over the 400 posterior targets. | yes |
| Multiple-cue judgment | 2.197 | Uniform over 9 legal responses: log 9. | yes |
| Multi-attribute choice | 0.693 | Uniform over 2 legal responses: log 2. | yes |
| Probability coherence | 32.13 | Absolute loss averaged over all 101 legal integer reports for each of the 2,460 saved human targets. | yes |
| SHJ categories | 50.0 | Uniform over 2 legal responses: 1/2. | yes |
| Medin categories | 50.0 | Uniform over 2 legal responses: 1/2. | yes |
| Weather prediction | 50.0 | Uniform over 2 legal responses: 1/2. | yes |
| RLWM | 33.3 | Uniform over 3 legal responses: 1/3. | yes |
| Recent probes | 50.0 | Uniform over 2 legal responses: 1/2. | yes |
| WCST, prediction | 1.386 | Uniform over 4 legal responses: log 4. | yes |
| WCST, autonomous | 25.0 | Expected accuracy of uniform card choices: 1/4. | no, analytic |
| Restless bandit | 50.0 | Expected better-arm rate of uniform choices between 2 arms: 1/2. | no, analytic |
| Instrumental learning | 75.0 | Credits both actions on tied-mean trials, over the original trial-weighted scorer. | no, analytic |
| Horizon, prediction | 0.693 | Uniform over 2 legal responses: log 2. | yes |
| Horizon, autonomous | 50.0 | Expected optimal rate of uniform choices on the 34,720 original free choices. The two arm means differ on every one. | no, analytic |
| Stationary bandits | 12.5 | Tied-maximal-arm probability under uniform actions, weighted over all 312,000 trials and 16 original conditions. | no, analytic |
| Hybrid sampling | 1.386 | Uniform over 4 legal responses: log 4. | yes |
| Mouselab, prediction | 1.634 | Log of the legal-option count, averaged over the 20,363 original held-out decisions. | yes |
| Mouselab, autonomous | 7.01 | Exact expectation over the 1,296 reachable known-node sets under uniform legal reveal and STOP choices, with the original independent four-point node prior and zero observation cost. A generating-distribution expectation, not a score on the models' sampled trees. No Monte Carlo rollouts. | no, analytic |
| BART | 0.84 | Exact expected reward of independent 1/2 skip and inflate decisions under the original hazard 1/(M - successful pumps), averaged over M = 128, 32 and 8. | no, analytic |
| Temporal discounting | 50.0 | Equal probability of either legal option on each realized reward question. A preference summary, not an accuracy. | no, analytic |
| Two-step | 0.50 | Uniform first-stage choices put equal mass on both states, and uniform second-stage choices yield the mean of the four current reward rates, on the identical 2,000-trial seeded rate schedule of the five open models. | no, analytic |
| False-belief QA | 50.0 | Uniform over the original 2 visible-container choices: 1/2. | yes |
| Politeness | 1.014 | Log of the legal-option count, weighted by original human response counts. | yes |
| Pragmatic interpretation | 1.282 | Log of the legal-option count, weighted by original human response counts. | yes |
| Nonliteral number | 34.77 | Continuous uniform report on [0, 100]. The expected absolute error for human mean h is (h^2 + (100 - h)^2) / 200, averaged over the 330 human means. | yes |
| Self-assessment | 2.039 | Pooled strict NLL on each question's complete support: 5 answers, 21 count reports, 101 percentile reports and 11 difficulty reports. Not the NLL of an eleven-bin display. | yes |

### Exact expectations

For a BART balloon with maximum M, the probability of surviving k inflations is (M-k)/M. Uniform skip or inflate therefore has expected reward sum over k from 0 to M-1 of k 2^-(k+1) (M-k)/M, averaged over M = 128, 32, and 8. In Two-step, uniform first-stage choices put equal mass on both states, and uniform second-stage choices yield the mean of the four current reward rates. All five original models retain the identical 2,000-trial seeded rate schedule, whose conditional expected reward is 0.496289. The continuous-report controls sample uniformly on [0, 1] for urn posteriors and [0, 100] for nonliteral percentages. Their exact MAE expectations are (p^2 + (1-p)^2)/2 and (h^2 + (100-h)^2)/200, averaged over the original 400 posterior targets and 330 human means.

Under uniform card choice, WCST perseveration and set loss are 17.13% each. These are exact expectations of `task_metrics` over the rule sequences of the 75 test participants, and a sampled check over 400 runs gives 17.12% and 17.13%. The same participants score 50.00% accuracy, 7.32% perseveration and 1.88% set loss.

Horizon uniform mean reward is 50.0 in every readout, from the arm means averaged over all 34,720 original free choices.

## Bayesian urn

The Bayesian urn reference is the analytic solution. Exact visible-information Bayes has posterior MAE zero by construction, and `examples/bayes_event_agent.py` computes it through the original scorer.

## RLWM and Horizon

Table 3 prints the argmax accuracy of the archived RLWM fit, the metric of its column. It is a derived readout, not generated author actions, and the mean correct-action probability of the same fit stays a separate row. The autonomous Horizon reference is printed under its softmax sampling rule, and a separate row holds the greedy replay of the same fit on the same 31 timelines.

## False-belief QA

ToMi has no human responses, so the reference is the answer key itself. It is 100 by definition, not a measured performance.

## References for the remaining Table 3 rows

The references marked ¶ in Table 3 are computed on CPU from the saved source data and original environments, with no neural or API call. Each uses the exact targets, aggregation, and scorer of the five open models. Recomputing the Uniform value and all five model scores through the same loaders reproduces Table 3. The type human marks references built from other participants' responses without a fitted model.

### Human-response references

Probability coherence predicts each of the 2,460 test responses with the median report of the 128 Psych-101 training participants for the same query in the same experiment. Self-assessment follows the lookup-table family of Binz et al. (2025) on the original integer supports. Each item has a response-frequency table with add-α smoothing, and α is chosen by participant cross-validation within the fitting data. Grammar tables are fitted on the 3,192 training sessions and scored on the 323 test sessions. The source defines no baseline for the logic set, so logic uses tenfold cross-fitting across logic participants. Transferring the grammar tables to logic instead gives 1.917.

Politeness and pragmatic interpretation predict each human response from the other participants' responses to the same item with add-one smoothing. Nonliteral number compares the mean reports of two random halves of each item's respondents, averaged over 1,000 seeded splits (SD 0.22). A half-panel mean lies about twice as far from the other half as from the full-panel mean, so this value is a conservative human reference.

Temporal discounting reports the mean later-reward rate of 13,232 Psych-101 participants on the same instrument under the realized-question rule of the model rows. That rule uses the actual selected option, excludes pure payment questions, and keeps all realized reward and speedup questions. Human sessions keep four anomaly items that the CogBench path skips after three immediate choices. Dropping them gives 39.8%.

### Autonomous references

Instrumental learning uses the CogBench Rescorla-Wagner model with separate learning rates for positive and negative prediction errors (Coda-Forno et al., 2024; Lefebvre et al., 2017). Its parameters are selected by mean reward on 100 independent training environments and frozen before the ten original test environments. BART uses one fixed pump threshold per balloon letter, selected the same way. The original runner reassigns the three maxima to the letters in every run, so these visible-information thresholds stay below the full-information optimum of 14.00, which requires the hidden maxima.

Two-step fits the seven-parameter hybrid model of Daw et al. (2011) by maximum likelihood to each of the 40 CogBench human participants. Every fitted parameter set then runs on the identical seeded reward-rate path of the five models, and the reported value averages the 40 runs. Twenty-five of the 40 fits have at least one parameter at a bound, so the fitted parameters are reported as a reference policy rather than as estimates of individual learning.

## New autonomous traditional-policy references

The Reference entries marked ∗ in Table 3 are newly executed adapted policy comparisons. Their records retain source identities, visible feedback, executed actions, and the original outcome scorers. They do not change the five local models' results or the separate human-prediction targets.

### Mouselab MetaGreedy

The original MetaGreedy policy family follows Callaway et al. (2022). We reuse all 95 archived participant-specific full fitted parameter sets without fitting on the generated evaluation trees. For each visible belief, the policy computes one-step expected value of information over the original equiprobable node values {-10, -5, 5, 10}. Its original terminate and select softmaxes and lapse mixture allocate probability only to STOP and legal frontier reveals. Fitted policy cost remains an internal termination parameter, and environment observation cost remains zero. Hidden node values enter the policy only after a selected reveal.

The Python action distribution is checked against all 20,363 saved original Julia cross-validation probability vectors, with maximum absolute difference 2.22×10^-16. Autonomous execution then uses the same 23,750 archived synthetic trees, 95-by-25-by-10 schedule, legal reveals, and belief-based STOP score as the five local models. The policy sampler uses an explicit per-episode NumPy seed. Its distribution matches the original policy, but it does not reproduce Julia's random stream bit for bit. The 95 identifiers denote MDP schedule slots for this generated evaluation, not 95 human behavioral trajectories. All 164,765 selected actions, visible beliefs, reveal feedback, and terminal scores are checked against the saved source trees. Mean terminal expected return is 11.7722, with 5.9375 reveals per episode. This is distinct from the archived human-prediction NLL of 1.3670.

### Restless Q-learning

We implement the original two-parameter Q-learning choice equations in the retained CogBench environment (Coda-Forno et al., 2024), using the authors' choice-function code at [Ershadmanesh/meta-meassure](https://github.com/Ershadmanesh/meta-meassure) (commit `eef1638`). Both initial values are 0.5. A chosen value updates as Q(a) ← Q(a) + α(r − Q(a)), with softmax inverse temperature β. As in the original choice model, rewards are expressed in units divided by 100. Updates receive only the integer reward shown in the original model-facing history. Hidden reward means and switch times enter scoring, never the policy.

Before the original test runs, the training protocol fixes 100 independent 80-trial environments (seeds 30000000 to 30000099), 101 evenly spaced α values in [0, 1], and β = 0 plus 81 logarithmically spaced values from 0.01 to 100. All 8,282 combinations are evaluated on the same training environment and policy random draws. Training mean earned reward selects α = 0.44 and β = 100, which are frozen before test execution. This β is the largest grid value. The best training reward over α rises from 56.74 at β = 89.1 to 56.83 at β = 100. This independent-simulation selection differs from the source paper's human-response fitting protocol, and no human-fitted result is claimed.

The test retains the ten original environment seeds 20261009 to 20261018, 80 trials per run, and block lengths 21, 20, 20, and 19. The original post-step better-arm accuracy convention is preserved, including switch-boundary trials. An original-environment replay exactly matches all 800 archived source rewards, means, and accuracy labels, and the saved test probabilities and updates also match the original choice functions on all 800 steps. The new reference scores 550/800, or 68.75% (68.8% at Table 3 precision), with mean reward 54.6679. No confidence mapping is fitted, and softmax choice probabilities are not interpreted as metacognitive confidence.

## Recovered references for the existing Table 3 targets

The references marked † are existing fitted or executed outputs, not new model calls. Their scored IDs, participant identity, human targets, objective labels, prefix boundaries, legal supports, and original aggregation have been aligned to the current neural comparison. Five fitted references cover the exact current targets for every frozen model. The stationary reference combines the existing eight Bernoulli Thompson conditions and the previously accepted eight Gaussian Thompson conditions, matching all 16 environment manifests, 30 seeds per condition, and 312,000 steps. No neural checkpoint, environment rollout, fitting objective, or response metric is changed.

| Current target | Reference | Score (N) | Fit, information, and implementation provenance |
| --- | --- | --- | --- |
| Multi-attribute human NLL | WADD | 0.1922 (576) | Four-weight paper-form reimplementation; 7,008 responses from 73 training participants; six held-out participants. |
| SHJ objective accuracy | GCM | 93.8319% (3,664) | Paper-form reimplementation; 29,776 responses from 85 training participants; ten held-out participants. |
| Medin objective accuracy | GCM | 96.6190% (6,300) | Train-only refit on 34,200 learning responses across the two source experiments; 42 held-out participant sessions. |
| Weather objective accuracy | GCM | 78.2500% (400) | Paper-form reimplementation; 4,600 responses from 23 training participants; two held-out participants. |
| Recent-probes accuracy | Rational model | 100.0% (3,774) | Conditional-response matrix fitted on 34,714 responses from 471 training participants; 51 held-out participants. |
| Stationary optimal-action rate | Thompson family | 68.0904% (312,000) | Official Beta(1,1) Thompson on Bernoulli rewards; disclosed normal-inverse-gamma Thompson on Gaussian rewards. Same original conditions, horizons, and environment seeds; separate logged action RNGs. |

These are existing references aligned to the primary targets of Table 3. Fits use source training participants and are reimplementations, not exact task-specific author executables. Classification accuracy compares each saved legal argmax to the original objective label under human history. It is not a human-likelihood or autonomous-reward score.

WADD, GCM, and the rational conditional-response model follow the published model families (Binz et al., 2025). Their original 1,000-update schedule-free fits retain the disclosed initialization, feature encoding, and learning rate. Medin's categorical model covers all 6,300 learning classifications used by the current accuracy row. Its separate 336 no-feedback classifications and 336 typicality reports remain in the archive. The typicality distribution is unsupported and the full 6,972-response human-likelihood task is not claimed complete. The Table 3 reference uses the local train-only refit, not the separately archived author-parameter replay. Recent probes supplies the current letter set in the prompt. Its fitted model selects the correct visible-set membership category on every item, and this does not demonstrate an internal working-memory mechanism.

The Thompson policies observe only their own past selected arms and rewards. Latent means enter the original optimal-action scorer, not the policies. The full 16-condition rate is step-weighted, matching Table 3, and no reward units are pooled. These are the original public FullBench conditions used for the neural adapters, not a claim of reproducing the publication's different horizon and seed protocol.

## Further reference details

The independent Hybrid ANN retains 20 participants, 163,135 responses, 160 folds, and 3,000 updates per fold.

Gaussian Thompson retains eight Gaussian conditions, 240 trajectories, and 156,000 steps. It uses a normal-inverse-gamma prior (μ0, κ0, α0, β0) = (0, 1, 1, 1) fixed before execution. Its policy sees its own rewards and arm counts, and latent arm parameters enter scoring only. The original eight Gaussian conditions and 30 seeds per condition are retained. Independent acceptance replays the original environment and verifies all posterior updates, actions, rewards, and scores.

The disclosed multiple-cue reference obtains held-out NLL 2.0291 over all 6,136 test responses. It preserves all 52,464 training responses from 232 participants and 6,136 test responses from 28 participants across three original experiments. Parameters are fitted separately by experiment with 27 deterministic starts, and all converge. The printed signed update formula is retained, the common softmax offset that cancels is fixed to zero, and updates occur only after participant-visible feedback. This is a disclosed reimplementation because the author fitting executable was unavailable.

The fitted MetaGreedy human-prediction reference has NLL 1.3670 over 20,363 actions, which is separate from autonomous return. The fitted Horizon RW uses three parameters under the retained data and feedback rules, and the different six-parameter formulation is not evaluated here.

## References

- Marcel Binz, Elif Akata, Matthias Bethge, and others. 2025. A foundation model to predict and capture human cognition. Nature, 644(8078):1002-1009.
- Frederick Callaway, Bas van Opheusden, Sayan Gul, Priyam Das, Paul M. Krueger, Thomas L. Griffiths, and Falk Lieder. 2022. Rational use of cognitive resources in human planning. Nature Human Behaviour, 6(8):1112-1125.
- Julian Coda-Forno, Marcel Binz, Jane X Wang, and Eric Schulz. 2024. CogBench: a large language model walks into a psychology lab. In Proceedings of the 41st International Conference on Machine Learning, PMLR 235:9076-9108.
- Nathaniel D. Daw, Samuel J. Gershman, Ben Seymour, Peter Dayan, and Raymond J. Dolan. 2011. Model-based influences on humans' choices and striatal prediction errors. Neuron, 69(6):1204-1215.
- Germain Lefebvre, Maël Lebreton, Florent Meyniel, Sacha Bourgeois-Gironde, and Stefano Palminteri. 2017. Behavioural and neural characterization of optimistic reinforcement learning. Nature Human Behaviour, 1(4):0067.
