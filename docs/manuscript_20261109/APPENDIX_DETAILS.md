# Supplementary appendix details

These details accompany the second arXiv version of the S1CogBench paper ([arXiv:2611.00911v2](https://arxiv.org/abs/2611.00911v2)). They preserve the complete tables and passages moved out of the PDF on 2026-11-09. Moving them does not change experiments, samples, scores or comparisons. Table names are used because manuscript numbers change.

- [API time and cost](#api-time-and-cost)
- [Task sources and coverage](#task-sources-and-coverage)
- [Full reference catalogue](#full-reference-catalogue)
- [Reference construction](#reference-construction)
- [Common-source comparison](#common-source-comparison)
- [Model and API protocol details](#model-and-api-protocol-details)
- [Matched one-token timing](#matched-one-token-timing)
- [Further related work](#further-related-work)

## API time and cost

| Model | Targets | Decisions | Median latency (s) | P95 latency (s) | USD per 1000 | Output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Jev | 27 | 864,813 | 0.35 | 0.41 | 0.0448 | 94 |
| Haiku | 15 | 92,803 | 2.60 | 7.65 | 0.336 | 442 |
| Sonnet | 3 | 2,330 | 3.08 | 4.99 | 5.39 | 253 |
| Opus | 2 | 730 | 3.59 | 5.34 | 8.96 | 289 |

API time and cost on each model's completed targets. Latency is the client round trip, so with different serving paths and targets it does not compare model speed. Output tokens count Jev's returned option probabilities and Claude's text with thinking.

[CSV copy](appendix_details/api_time_cost.csv). The detailed per-target records are in [docs/API_TIME_COST.md](../API_TIME_COST.md) and [results/aggregate/api_time_cost_per_target.csv](../../results/aggregate/api_time_cost_per_target.csv).

## Task sources and coverage

| Task mode | Task source | Reference type | Mark | Haiku |
| --- | --- | --- | --- | --- |
| Multiple-cue judgment | [Collsiöö et al., 2023](https://doi.org/10.1016/j.cognition.2023.105584) | human fit | -- | n.a. |
| Multi-attribute choice | [Hilbig et al., 2014](https://doi.org/10.3758/s13423-014-0643-0) | human fit | † | -- |
| WCST, prediction | [Grant et al., 1948](https://doi.org/10.1037/h0059831), [Namazova et al., 2025](https://arxiv.org/abs/2508.07887v1) | human fit | -- | n.a. |
| Horizon, prediction | [Wilson et al., 2014](https://doi.org/10.1037/a0038199), [Namazova et al., 2025](https://arxiv.org/abs/2508.07887v1) | human fit | -- | -- |
| Hybrid sampling | [D'Ambrogio et al., 2026](https://doi.org/10.1038/s41593-026-02342-9) | human fit | ‡ | n.a. |
| Mouselab, prediction | [Callaway et al., 2022](https://doi.org/10.1038/s41562-022-01332-8) | human fit | ‡ | n.a. |
| Politeness | [Yoon et al., 2016](https://escholarship.org/uc/item/6dm139m8) | human | ¶ | -- |
| Pragmatic interpretation | [Hu et al., 2023](https://doi.org/10.18653/v1/2023.acl-long.230) | human | ¶ | -- |
| Self-assessment | [Jansen et al., 2021](https://doi.org/10.1038/s41562-021-01057-0) | human fit | ¶ | n.a. |
| SHJ categories | [Badham et al., 2017](https://irep.ntu.ac.uk/30691/1/8441_Badham.pdf) | human fit | † | -- |
| Medin categories | [Levering et al., 2020](https://doi.org/10.3758/s13421-019-00972-y) | human fit | † | -- |
| Weather prediction | [Speekenbrink et al., 2008](https://discovery.ucl.ac.uk/id/eprint/14463/) | human fit | † | -- |
| RLWM | [Collins et al., 2012](https://doi.org/10.1111/j.1460-9568.2011.07980.x) | human fit | § | -- |
| Recent probes | [Enkavi et al., 2019](https://doi.org/10.1073/pnas.1818430116) | human fit | † | -- |
| False-belief QA | [Le et al., 2019](https://doi.org/10.18653/v1/D19-1598) | optimal | -- | -- |
| WCST, autonomous | [Grant et al., 1948](https://doi.org/10.1037/h0059831), [Namazova et al., 2025](https://arxiv.org/abs/2508.07887v1) | human fit | -- | n.f. |
| Restless bandit | [Coda-Forno et al., 2024](https://proceedings.mlr.press/v235/coda-forno24a.html) | algorithm | * | part. |
| Instrumental learning | [Lefebvre et al., 2017](https://doi.org/10.1038/s41562-017-0067) | algorithm | ¶ | part. |
| Horizon, autonomous | [Wilson et al., 2014](https://doi.org/10.1037/a0038199), [Namazova et al., 2025](https://arxiv.org/abs/2508.07887v1) | human fit | -- | n.f. |
| Stationary bandits | [Nie et al., 2025](https://proceedings.mlr.press/v267/nie25b.html) | algorithm | † | n.e. |
| Mouselab, autonomous | [Callaway et al., 2022](https://doi.org/10.1038/s41562-022-01332-8) | human fit | * | n.e. |
| BART | [Frey et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5627985/) | algorithm | ¶ | -- |
| Two-step | [Daw et al., 2011](https://doi.org/10.1016/j.neuron.2011.02.027) | human fit | ¶ | -- |
| Bayesian urn | [Coda-Forno et al., 2024](https://proceedings.mlr.press/v235/coda-forno24a.html) | optimal | -- | -- |
| Probability coherence | [Zhu et al., 2020](https://doi.org/10.1037/rev0000190) | human | ¶ | part. |
| Nonliteral number | [Kao et al., 2014](https://doi.org/10.1073/pnas.1407479111), [Tsvilodub et al., 2025](https://escholarship.org/uc/item/08c3s3jt) | human | ¶ | -- |
| Temporal discounting | [Coda-Forno et al., 2024](https://proceedings.mlr.press/v235/coda-forno24a.html) | human | ¶ | -- |

Task sources, references in the manuscript's main-results table, and empty Haiku cells. Types are fits to participants (human fit), other participants' responses (human), analytic solutions or answer keys (optimal), and policies without human fitting (algorithm). Marks are an existing fit or policy (†), a same-participant fit (§), a within-participant holdout (‡), an adapted policy (*), and a value computed from source data (¶). The reference-construction section below explains †, *, and ¶. Haiku cells are n.e. (not measured), n.a. (no full probability score), n.f. (partial prefixes only), or part. (incomplete records).

[CSV copy](appendix_details/task_sources.csv).

## Full reference catalogue

| Group | Reference | Data used | Construction |
| --- | --- | --- | --- |
| Selected references | Exact Bayes | Generated urn trials, no human fitting | Analytic posterior |
| Selected references | Horizon RW fit | 25 training participants, 6 held out | Three-parameter fit, autonomous sampling with source seeds |
| Selected references | WCST SL refit | 300 training participants, 75 held out | Refit with separate prediction and autonomous histories |
| Selected references | Mouselab MetaGreedy | 95 participants, 5 folds | Human-action fit using official code, 100 Sobol starts |
| Selected references | Hybrid independent ANN | 20 participants, 160 validation folds | Independent training, 3,000 updates per fold |
| Selected references | Multiple-cue update model | 232 training participants, 28 held out | Source-formula reimplementation, updates only from visible feedback |
| Selected references | Gaussian Thompson | 8 conditions, 30 seeds each, no human fitting | NIG prior and posterior, original environment and scoring |
| Selected references | RLWM parameters | Archived participant fits | Replay of fitted parameters |
| Computed from source data (¶) | Probability coherence | 128 Psych-101 training participants | Median response to the same query |
| Computed from source data (¶) | Self-assessment | Source lookup table ([Binz et al., 2025](https://doi.org/10.1038/s41586-025-09215-4)) and logic-item responses | Add-α smoothing, tenfold cross-fitting across participants for logic items |
| Computed from source data (¶) | Politeness, pragmatic interpretation | Other participants' responses to the same item | Response prediction with add-one smoothing |
| Computed from source data (¶) | Nonliteral number | Two random halves of each item's respondents | MAE between the two mean reports |
| Computed from source data (¶) | Temporal discounting | 13,232 Psych-101 participants | Mean later-reward choice rate on the same questions |
| Computed from source data (¶) | Instrumental learning | 100 independent training environments for parameter selection | CogBench Rescorla-Wagner, separate rates for positive and negative prediction errors, fixed before test ([Coda-Forno et al., 2024](https://proceedings.mlr.press/v235/coda-forno24a.html), [Lefebvre et al., 2017](https://doi.org/10.1038/s41562-017-0067)) |
| Computed from source data (¶) | BART | Training environments for threshold selection | One fixed pump threshold per balloon letter |
| Computed from source data (¶) | Two-step | 40 CogBench participants | Seven-parameter hybrid fit to each participant ([Daw et al., 2011](https://doi.org/10.1016/j.neuron.2011.02.027)) |

Reference data and construction. Fitting data and evaluation targets differ across references, so their training budgets are not directly comparable.

RLWM replays archived participant fits and is not a held-out comparison with the models.

The self-assessment source has no logic-item baseline, so those items use tenfold cross-fitting across participants.

Nonliteral number uses only half the respondents for each mean, making the reference conservative.

BART thresholds remain below 14.00, the optimum when the hidden maxima are known.

In Two-step, 25 of the 40 fits have at least one parameter at a bound. They serve as a reference policy, not as estimates of individual learning.

[CSV copy](appendix_details/reference_catalogue.csv). The manuscript retains the selected references and their main comparability distinctions.

## Reference construction

References marked † are existing model families fitted to the source training participants and reimplemented here, namely WADD for multi-attribute choice, GCM for SHJ, Medin, and weather prediction, the rational conditional-response model for recent probes, and the Thompson family for stationary bandits ([Binz et al., 2025](https://doi.org/10.1038/s41586-025-09215-4)). References marked * adapt existing policies. Mouselab MetaGreedy reuses 95 archived participant parameter sets ([Callaway et al., 2022](https://doi.org/10.1038/s41562-022-01332-8)), and Restless runs a two-parameter Q-learning model whose parameters are selected on 100 training environments, with β at the edge of its grid ([Coda-Forno et al., 2024](https://proceedings.mlr.press/v235/coda-forno24a.html)). References marked ¶ are computed on CPU from the saved source data with the models' targets, aggregation, and scorer. The full reference catalogue above summarizes their data and construction. The recent-probes prompt shows the current letter set, so its fitted model answers every item correctly, and the stationary bandits use the public FullBench conditions rather than the publication's protocol. The supplement's `docs/PAPER_DETAILS.md` and `docs/BASELINES.md` give each construction in full.

## Matched one-token timing

For each checkpoint, the first 200 saved requests of six input modes are timed in pairs, a direct legal-choice forward pass against `generate(max_new_tokens=1)` with greedy decoding and no legal-token mask, on the same prompt at batch size one in BF16 on one A100-SXM4-40GB. The generation path adds little overhead, but 524/1,200 Llama and 936/1,200 Qwen outputs fall outside the legal token set, so one unconstrained token is not a complete answer protocol.

| Model | Input mode | Forward P50 (ms) | Forward P95 (ms) | One-token P50 (ms) | One-token P95 (ms) | Paired ratio median | Invalid out of 200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | Horizon | 36.791 | 37.437 | 37.540 | 38.206 | 1.0202 | 0 |
| Llama | Recent probes | 122.968 | 218.740 | 123.929 | 219.720 | 1.0061 | 78 |
| Llama | Multi-attribute | 152.476 | 253.644 | 153.238 | 255.317 | 1.0049 | 44 |
| Llama | RLWM | 51.483 | 97.594 | 52.249 | 98.330 | 1.0145 | 200 |
| Llama | SHJ | 271.889 | 600.766 | 269.462 | 570.016 | 1.0024 | 2 |
| Llama | WCST | 243.577 | 477.837 | 244.291 | 480.547 | 1.0036 | 200 |
| Qwen | Horizon | 50.270 | 50.625 | 51.073 | 51.475 | 1.0155 | 194 |
| Qwen | Recent probes | 91.552 | 152.456 | 92.322 | 151.815 | 1.0079 | 3 |
| Qwen | Multi-attribute | 108.573 | 186.266 | 109.324 | 187.203 | 1.0070 | 144 |
| Qwen | RLWM | 51.233 | 75.347 | 51.947 | 77.273 | 1.0138 | 200 |
| Qwen | SHJ | 207.367 | 418.420 | 208.147 | 419.261 | 1.0043 | 195 |
| Qwen | WCST | 176.393 | 336.146 | 177.112 | 337.135 | 1.0053 | 200 |

Matched one-token timing control, with 200 pairs in every row. The ratio column is the median of within-request generation/forward ratios, not the ratio of the two marginal medians. Invalid counts are first tokens outside the legal option-token set, such as a newline, punctuation, or the start of a longer answer, not complete task errors.

[CSV copy](appendix_details/one_token_timing.csv). The per-request timing records and the CPU source verification are not included in this repository.

## Common-source comparison

Saved decisions on common source identities, 200 Horizon free choices from one timeline and 74 WCST decisions from two participants. Claude answers in a final line of text with adaptive thinking, so the readouts are not matched, and Haiku's two invalid WCST attempts count as wrong. The seven open models are compared under legal argmax and categorical sampling, each retaining its own action-dependent history.

### Open models under shared legal readouts

| Model | Horizon argmax reward | Horizon sampling reward | WCST argmax correct (%) | WCST sampling correct (%) |
| --- | --- | --- | --- | --- |
| Laya | 50.23 | 50.05 | 20.27 | 22.97 |
| Intern | 56.22 | 51.35 | 33.78 | 29.73 |
| Llama | 54.33 | 51.57 | 37.84 | 39.19 |
| Minitaur | 56.48 | 52.92 | 43.24 | 27.03 |
| Qwen | 52.68 | 51.33 | 18.92 | 22.97 |
| Nox | 56.62 | 54.94 | 28.38 | 22.97 |
| CLM | 49.58 | 49.88 | 25.68 | 32.43 |

### Claude under adaptive free-text readout

| Model | Horizon reward | WCST correct (%) |
| --- | --- | --- |
| Opus | 56.87 | 63.51 |
| Sonnet | 56.95 | 66.22 |
| Haiku | 56.86 | 66.22 |

[CSV copy](appendix_details/common_source_comparison.csv). Free-text results are stored in their own CSV columns. They are not duplicated as legal-argmax or sampling measurements.

## Model and API protocol details

### Nox and CLM

Nox is Decision-2.0-Nox-4B at revision ae269ec, run through its own runtime with a 16,384-token context, so no input is truncated. CLM is CLM-v0.1-8B at revision e939398 with the official vLLM pooling server for Qwen3-8B. That setup keeps the first 2,048 tokens of each text, so in 11 modes it cuts the end of long states, including the question, on some calls, such as 66% of the original WCST calls. A control that keeps the last 2,048 tokens instead changes the scores but leaves every comparison with uniform choice unchanged (`docs/PAPER_DETAILS.md` in the supplement). Under its original native readout, CLM always chooses machine H in autonomous Horizon and always stops in autonomous Mouselab, so it scores exactly 50% and zero reward there.

### API models

Jev 1.14 is the `jev-1.14-20261014` snapshot. It executes its native choice, and its two-decimal probabilities are normalized without a floor. The Claude5.5 models use adaptive thinking, medium effort, and a 4,096-token cap and answer on a final line of text, and an invalid answer counts as wrong and ends a closed-loop run. Haiku's NLL uses probabilities it states in JSON, a different readout from the open models.

Several stated politeness probabilities closely match the published human proportions, so exposure to these public data cannot be excluded. Opus and Sonnet score 0.001 on Bayesian urn and 15.34 and 20.81 on nonliteral number, and Sonnet scores 91.1% on Restless.

## Further related work

### Cognitive benchmarks

CogToM builds a theory-of-mind benchmark from 46 paradigms ([Tong et al., 2026](https://aclanthology.org/2026.acl-long.1448/)), and EPITOME compares five language models with human baselines on six theory-of-mind experiments ([Jones et al., 2024](https://aclanthology.org/2024.tacl-1.45/)).

### Predicting human behavior

Cross-Centaur predicts each participant's decisions in one of 12 psychological tasks from the same participant's behavior in another ([Asano et al., 2026](https://aclanthology.org/2026.emnlp-main.2235/)), and SocSci210 finetunes language models on 170 of 210 social science experiments to predict human response distributions ([Kolluri et al., 2025](https://aclanthology.org/2025.emnlp-main.1530/)). [Sun et al., 2026](https://aclanthology.org/2026.findings-emnlp.2177/) find that context shifts LLM product choices in human-like ways but does not reliably reweight attention between features.

### Sequential decisions

Horizon manipulates future opportunities and information asymmetry ([Wilson et al., 2014](https://doi.org/10.1037/a0038199)), and BanditBench supports interactive choices with raw or summary histories and regret analysis ([Nie et al., 2025](https://proceedings.mlr.press/v267/nie25b.html)). Mouselab and Tool-Lab study information acquisition and its consequences ([Callaway et al., 2022](https://doi.org/10.1038/s41562-022-01332-8), [Wadi et al., 2026](https://arxiv.org/abs/2609.28372v2)).

### Other System One evaluations

[Rafe et al., 2026](https://arxiv.org/abs/2610.00346v2) compare decision models with classifiers and general language models. DecisionPivot diagnoses probability ranking, calibration, and selective coverage with controlled task families ([Zhuang et al., 2026](https://arxiv.org/abs/2610.28522v1)), and DecisionRobust studies text and vision perturbations and prompt injection ([Lan et al., 2026](https://arxiv.org/abs/2610.19118v1)).
