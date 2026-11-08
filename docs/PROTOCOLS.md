# Evaluation protocols

This document retains the released arXiv snapshot. Use [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) for current CLI action-rule behavior and [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the separate current-paper evidence. `plan --json` exposes the same original mode metadata used by `eval`.

The released arXiv snapshot has 24 tasks and five frozen open models. All 120 task–model pairs have an accepted full-sample named protocol. The catalogue retains 31 paradigms in ten domains, including seven deferred tasks. Completing a compact presentation or human-response mode does not complete all original presentations or autonomous modes.

## Preserve the evaluation unit

Human-response prediction conditions on the participant's preceding actions and observations. Autonomous evaluation uses the model's own actions and resulting feedback. An explicit event report has a named event and observed label. Their likelihoods, rewards, and calibration measures answer different questions.

Retain the original participant split or folds and complete test denominator. Within-participant learning/test phases differ from participant holdouts. Frozen checkpoints, training-participant-fitted cognitive references, and same-participant archived-parameter replays have different fitting exposure.

## Presentation and readout

Compact multiple-cue histories retain all source events, order, responses and visible feedback. Compact variants in other catalogue tasks remain separately labeled; they do not complete original full-text presentations. Native distributions, legal-option likelihood, full-vocabulary likelihood, integer-sequence likelihood, native argmax, and sampled generation retain separate metadata. Intern's numerical readout retains the complete legal integer sequence. Horizon and WCST retain native argmax or source sampled-generation rules per model, and these readouts are identified with every protocol. Complete human histories provide external information support and do not directly measure human working-memory capacity.

Grammar retains both original 11-bin and full-integer NLL on the same 9,044 test responses. Binning sums saved legal-integer probabilities. It does not rerun the model or replace the integer score.

Horizon uses SDK probability sampling for Laya, source sampled-token rules for Llama and Minitaur, and native legal argmax for Intern and Qwen. WCST uses legal argmax for Laya/Intern/Qwen; the two 8B models use full-vocabulary next-token argmax in prediction and sampled generation in autonomous execution.

WCST autonomous rollouts replay source switch times. All 3,059 switches follow criterion human performance; model actions do not cause the switches. RLWM histories provide external information support. Hybrid uses an author posterior summary. Autonomous Mouselab has zero observation cost and terminal selection by expected value. Stationary bandits use the summary-history context of the official code in every accepted row, which gives counts and mean rewards from all prior feedback. `scripts/run_bandit_open_models.py --history raw` runs the raw-history context, and no raw-history result is accepted here.

## Scores and conditions

Use accuracy, original reward, pseudo-regret, MAE, human-response NLL, event or answer-category Brier, P50/P95, and invalid rate only where their targets apply. Class-summed and single-event Brier have different normalization. Confidence and action probabilities are not automatically success probabilities. Latency is not human reaction time.

Retain original effects such as horizon-dependent information choice, rule-switch errors, load, learning curves, and evidence sensitivity. Conditions, units, n, and aggregation stay attached. Infinite, undefined, unavailable, and not-applicable values are distinct. No heterogeneous reward total or global cognitive score is defined.

## Reference provenance

Analytic solutions, independent fits, author-code fits, disclosed reimplementations, and archived-parameter replays are distinct reference types. [BASELINES.md](BASELINES.md) and [baseline_provenance.json](../results/aggregate/baseline_provenance.json) record them per reference, with data splits, original scoring rules, and implementation deviations.

## Compare and reproduce

[Aggregate files](../results/aggregate/results.json) carry task, protocol, mode, metric variant, target, scope, unit, n, aggregation and notes. [Metric applicability](../results/aggregate/metric_status.json) retains unresolved additional modes without labeling them complete. [Reference provenance](../results/aggregate/baseline_provenance.json) distinguishes official code, disclosed reimplementation and archived replay.

Match the comparison context before grouping models. Timing additionally requires matching execution context. See [CPU validation](CPU_REUSE_VALIDATION_24.md) and [data requirements](DATA.md).

## Full protocol statement

### Models and histories

The frozen open-model comparison comprises Laya English and Intern-Decision-4B as structured decision models, Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as a general language-model control, Minitaur-8B (Binz, n.d.) as a human-behavior-trained control, and Qwen3.5-4B (Qwen Team, n.d.) as a backbone-family comparison for Intern. Same-family membership does not imply matched training data or compute. Jev is additionally measured in the [selected-task API extension](API_EXTENSION.md), separate from the complete five-model suite. Model roles, source implementations, and readouts are in [MODELS.md](MODELS.md).

Models receive the same task-visible facts and legal options under each specified human-history protocol. Autonomous evaluations instead supply each model's own action-dependent history under common environment rules. Original generative decoders, native legal choices, and externally sampled actions are labeled separately. A long reasoning trace is not required, and any additional reasoning budget is a distinct condition. Complete, summary, finite-window, and absent histories are different information regimes. Full-text context-overflow records are retained separately from completed compact protocols. The catalogue retains approved compact variants for multiple-cue judgment, reward revaluation, Columbia Card Task, and risky-choice information search, preserving all original events, order, and visible feedback. Only multiple-cue judgment enters this fixed 24-task evaluation, and the other three remain deferred catalogue records. Original and compact results stay separately identified.

### Three scoring targets

Human-response prediction evaluates the probability of the observed participant response given that person's preceding history, NLL = −(1/N) Σ log p(a_it | h_it) over participants i and scored trials t. Participant-, fold-, and response-weighted aggregation follow the stated source protocol. Legal-response NLL differs from full-vocabulary NLL. The typed track retains returned zeros as infinite NLL, and legacy clipping or additive constants are disclosed. A scalar report supports report error but does not specify a distribution over all possible reports. For example, predicting a human integer response from 0 to 100 requires 101 candidate probabilities for NLL. This is an added evaluation readout, not a requirement of the original experiment. Grammar additionally retains the original 11-bin response NLL by summing all legal-integer probabilities into the source bins, and full-integer NLL is kept alongside it.

Autonomous performance uses executed trajectories and the original reward and cost definitions, G = Σ_t (r_t − c_t) for return and R = Σ_t [max_a μ_t(a) − μ_t(a_t)] for pseudo-regret. Costs are applied once, and expected route values and realized returns remain separate. Pseudo-regret requires an explicit environment reference.

Event-probability evaluation requires an elicited probability q for a named binary event and its observed label Y, giving Brier error (q − Y)². Action probabilities and ordinal confidence do not automatically provide this target. For objectively labeled answer categories, the additional class-summed Brier score is Σ_k (p_k − 1[y = k])², and its normalization differs from single-event binary Brier. Posterior MAE, human-rating MAE, and correctness likewise retain their own targets and scales.

### Metrics and cognitive effects

The nine shared fields are accuracy, original reward or net reward, pseudo-regret, MAE, human-response NLL, event or answer Brier, latency P50 and P95, and invalid-response rate. [metric_status.json](../results/aggregate/metric_status.json) records their applicability, and incompatible units are not pooled into a cognition score. Original cognitive measures additionally examine evidence sensitivity, horizon-dependent exploration, rule-switch errors, learning load and intervals, stopping, and reward-transition effects. Observable condition effects do not uniquely identify internal mechanisms.

| Common field | Required target or reference | Reporting boundary |
| --- | --- | --- |
| Accuracy | Original objectively correct response | Agreement with a person is a different statistic |
| Reward / net reward | Executed trajectory and original payoff | Information costs included exactly once |
| Pseudo-regret | Defined environment reward reference | State stationary or time-varying reference |
| MAE | Numeric target on its source scale | Posterior error and human-rating error separate |
| Human-response NLL | Real observed response and visible history | Legal-set and full-vocabulary readouts separate |
| Event / answer Brier | Named event or objective answer label | Binary and class-summed normalization stated |
| Latency P50 | Declared request timing boundary | Inference-only and end-to-end values separate |
| Latency P95 | Same boundary as P50 | Input length, hardware, sharing and batching stated |
| Invalid-response rate | Original legal-response contract | Invalid outputs retained in source denominators |

Applicability is defined per task and protocol, and unavailable values and inapplicable metrics are distinct. Task-specific cognitive effects are reported in addition to these fields.

Human-prediction uncertainty uses participants, and autonomous uncertainty uses independent runs or environments. Steps within a trajectory are not independent replications. Source estimates are retained, with added paired 95% intervals and their resampling unit identified. References are labeled as analytic solutions, official-code fits, disclosed reimplementations, or replays of archived predictions or parameters. Their fitting objective and access to participants accompany the comparison, and same-participant fits and held-out neural prediction are distinct.

### Runtime and reproducibility

Runs identify checkpoints, adapters, readouts, precision, histories, hardware, and sharing context. P50 and P95 retain their timing definition and recorded batching conditions. Shared-device latency does not establish an isolated speed ranking or measure human reaction time. The additional controls use all five models on the original Horizon and WCST inputs under both legal argmax and sampling, retaining each model's prompt family, precision, kernel, history, and source seeds. Same-A100 batch-one latency uses 20 warm-up calls per mode and reports nine implementation configurations separately. See [SUPPLEMENTARY_CONTROLS.md](SUPPLEMENTARY_CONTROLS.md), [LATENCY.md](LATENCY.md) and [CPU_REUSE_VALIDATION_24.md](CPU_REUSE_VALIDATION_24.md).

## References

- Marcel Binz. n.d. Llama-3.1-Minitaur-8B: Model card. Hugging Face.
- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, and others. 2024. The Llama 3 herd of models.
- Qwen Team. n.d. Qwen3.5-4B: Model card. Hugging Face.
