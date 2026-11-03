# Selected-task API extension

The API extension evaluates Jev and Claude 5.5 on the original complete sample of each completed target in Table 3. It does not add these models to the five-model suite, and the open-model results are unchanged. Requests reproduce the visible states, histories, legal options, environments, and seeds of the open-model runs. Closed-record replay verifies every saved request and response against the original sources and scorers, and no answered request is sent again.

Jev 1.14 completes all 27 modes. Claude Haiku 5.5 completes 15 modes, Claude Sonnet 5.5 completes 3 (Bayesian urn, Restless bandit, Nonliteral number), and Claude Opus 5.5 completes 2 (Bayesian urn, Nonliteral number). The rows are in [results.json](../results/aggregate/results.json) with `model_access` set to `api`. Time and cost per target, and the cells that were not completed, are in [API_TIME_COST.md](API_TIME_COST.md).

A cell without a value keeps its Table 3 marker.

| Marker | Meaning |
| --- | --- |
| n.e. | Outside the measured full-task scope |
| n.a. | Lack a full probability score |
| n.f. | Only non-full prefixes exist; see [SUPPLEMENTARY_CONTROLS.md](SUPPLEMENTARY_CONTROLS.md) |
| part. | Incomplete records are retained |
| canc. | Canceled before generation |

## Jev

OpenRouter routes `typesafe/jev-1.14` to the recorded `typesafe/jev-1.14-20261014` snapshot. The native choice is the executed action. The raw two-decimal option probabilities are divided by their returned sum when the sum lies within a 0.005 per-option rounding bound, and no probability floor is applied. The API exposes neither a generation seed nor numerical precision.

In RLWM, Table 3 prints the argmax accuracy of the fitted reference, 89.3% by responses, against 89.5% for Jev. By participant these are 89.28% (95% CI 88.96 to 89.59) and 89.37% (89.10 to 89.63), and mean correct-action probability is 80.09% against 88.22%. The autonomous Horizon reference is printed under its softmax sampling rule, 74.6%. Greedy replay of the same fit on the same 31 timelines gives 84.9% and mean reward 55.99, and Minitaur under legal argmax gives 84.0%. Other autonomous references keep their recorded sampling rules, whereas model readouts follow [PROTOCOLS.md](PROTOCOLS.md).

Table 3 uses each source scorer's own epsilon where its primary metric defines one, as for the open models. Strict NLL without any epsilon is infinite wherever Jev gives zero probability to a human choice. This occurs for 8 of 12,754 WCST responses, 353 of 6,136 multiple-cue items, 28 of 163,135 hybrid-sampling choices, 58 politeness and 103 pragmatic-interpretation items, and 3,611 of 108,248 self-assessment responses. The 10^-12 clipping of the multiple-cue and self-assessment scorers gives 3.069 and 2.724, while the hybrid-sampling and social scorers define no clipping. The two clipped values are separate rows with `metric_variant` set to `typed_choice_mean_nll_clipped_1e12`.

Jev's multiple-cue row uses the full original prefix, so its `protocol` is `original_full_prefix_typed`, as for the accepted Llama row.

## Claude 5.5

The models are `claude-opus-5-5`, `claude-sonnet-5-5`, and `claude-haiku-5-5`, with adaptive thinking, summarized display, medium effort, and a 4,096-token output cap. No temperature, top-p, or top-k is sent. The answer is parsed from a single final line of the native text that must name one complete legal option key, and thinking is saved separately. A truncated answer or one without such a line is invalid and counts as wrong. In a closed-loop task an invalid answer ends its run, and a task with an incomplete run is marked part. This occurs in one of ten Restless runs and two of ten instrumental-learning runs for Haiku. Sonnet completes all ten Restless runs. Its separately elicited confidence reports are not used as action probabilities. Non-full Claude Horizon and WCST prefixes appear only in [SUPPLEMENTARY_CONTROLS.md](SUPPLEMENTARY_CONTROLS.md).

## Claude stated probabilities

Claude returns no option probabilities, so its human-response NLL is not available from choices. For politeness, pragmatic interpretation, multi-attribute choice, and Horizon prediction, Haiku instead states a probability for every legal option in a JSON object on its final line. These stated probabilities are a different readout from the option probabilities of the open models, and Table 3 marks them. They are normalized under Jev's rounding rule without a floor.

In politeness one of 128 responses is invalid because its probabilities sum to 0.67. Two valid responses give zero probability to a human choice, so the strict NLL is infinite under any treatment of the invalid response. Pragmatic interpretation has one such zero. In multi-attribute choice and Horizon prediction no human choice receives zero. In multi-attribute choice Haiku's most probable option matches the human choice in 97.7% of 576 decisions. Several stated politeness probabilities closely match the published human proportions, so exposure to these public data cannot be excluded.

## Run records

### Jev native decisions

Native choice labels and raw option probabilities are saved. Two-decimal rounding can yield a probability sum of 0.99 or 1.01. Original visible states, legal keys, source identities, own-action histories, feedback, environment seeds, and scorers are retained.

A closed-record source replay verifies the 400 Bayesian answers, ten 80-trial Restless runs with 800 actions and 800 separate confidence reports, all 6,720 Horizon human-prediction choices from six participants, and both 12,754-choice WCST modes from 75 participants. Bayesian posterior MAE is 0.068218. Original post-step Restless better-arm accuracy is 63.875%. WCST autonomous participant-mean accuracy is 28.4985%. Confidence is a separate native query about whether the selected machine currently has the higher average reward, rather than the action's choice probability. Its interpretation is disclosed separately from the original verbal confidence scale.

Horizon human-prediction NLL is 0.414095 under its original 10^-8 additive epsilon, and WCST human-prediction participant-mean NLL is 1.543327 under its original 10^-12 convention. These source-epsilon scores are the Table 3 targets. The saved native probabilities also preserve exact zeros. The strict native Horizon participant-mean NLL is 0.414095, with 0 zero-probability human targets among 6720 responses. The strict native WCST participant-mean NLL is +∞, with 8 zero-probability human targets among 12754 responses.

Jev's complete Horizon autonomous evaluation retains all 31 source participants and 34,720 free choices. Its original mean optimal-choice rate is 82.9061%, and mean free-choice reward is 55.8734. Closed-source replay verifies the complete original states, own-action histories, source environment seeds, choices, feedback, and scoring. The saved prefix is reused without additional API calls. Four independent participant timelines are scheduled concurrently while each participant's trials remain sequential. Together, the four original tasks account for 68,948 verified responses and a known usage-priced cost of US$4.6634. The charge for a prior failed HTTP request remains unknown and is not included in that known total. These are usage prices rather than account invoices.

Jev also completes all 2,460 probability-coherence judgments from 15 original human histories and all 330 nonliteral-number queries. Probability coherence retains the original 101 integer response keys from 0 to 100 and scores the native choice against the source human report. Its original MAE is 17.655691 percentage points. Complete human prefixes are supplied unchanged, without replacing prior human reports with generated answers. Nonliteral-number evaluation uses the same 60 public scenarios and original slider scaling described below. Its MAE to the human mean is 20.372091. Both complete additions have zero invalid or truncated responses, and every saved native request, response, source identity, parser result, usage record, and original score passes closed-record verification. They raise the known cumulative Jev usage-priced cost to US$5.0465 for 71,738 responses, and the prior failed-request charge remains unknown.

The remaining Claude generation was canceled after Jev completed. Haiku retains 181 of 2,460 probability-coherence responses, all valid and source-verified. Sonnet and Opus were canceled before generation. No complete Claude probability-coherence metric is defined.

### Claude Bayesian inference

Each of the three models completes the exact same 400 generated Bayesian questions (200 original runs, source seed 20261007) used for the other Bayesian rows. The final numeric probability is parsed from native text. Each model has 400 valid answers, zero invalid answers, and zero truncated answers. All three have posterior MAE 0.000970843 at their returned precision. This is posterior error, not a claim of 100% event accuracy. The original Bayesian scorer and its condition-specific results remain in the record.

| Claude 5.5 | N | MAE | P50 (s) | P95 (s) | US$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus | 400 | 0.000971 | 3.501 | 4.446 | 3.0832 |
| Sonnet | 400 | 0.000971 | 2.104 | 2.943 | 1.0524 |
| Haiku | 400 | 0.000971 | 2.215 | 4.380 | 0.0861 |

These are complete Bayesian API targets after original-source, identity, native request and response, parser, usage, and scoring verification. Latencies are non-streaming native message-request client round trips, including network and provider service time, excluding the preceding token-count request. Usage-priced costs are not account invoices. No API generation seed, dtype, action probabilities, time to first token, or provider-only forward time is inferred.

### Claude nonliteral-number interpretation

All three Claude 5.5 models complete the same 330 original slider questions from 60 public scenarios: 300 queries in experiment 1 and 30 in experiment 3b (Tsvilodub et al., 2025). All co-present questions and the complete source instructions remain visible. Each native scalar report in [0, 1] is multiplied by 100 for the original slider and original MAE-to-human-mean scorer. No normalization is imposed across separate questions, and no action probability vector or human-response NLL is constructed. These public materials have no official machine-learning test split, and no unseen-material claim is made. The original CogGym scorer and its experiment and condition breakdowns are retained. All 990 answers pass source, identity, native request and response, parser, usage, price, and original-score checks, with zero invalid or truncated responses.

| Claude 5.5 | N | MAE | P50 (s) | P95 (s) | US$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus | 330 | 15.3355 | 3.735 | 5.887 | 3.4542 |
| Sonnet | 330 | 20.8072 | 2.788 | 3.866 | 1.5156 |
| Haiku | 330 | 22.5211 | 3.019 | 5.586 | 0.0861 |

These are complete nonliteral-number API targets, using the original 0 to 100 scale and response-weighted MAE over all 330 queries. Latency and usage-price conventions match the Bayesian table. The three-model usage-priced total is US$5.0558, below the separate US$10 ceiling, and is not an account invoice. No failed response is retried.

### Claude Restless bandit

Sonnet completes all ten original 80-trial runs, with 800 actions and 800 separately elicited confidence reports. All 1,600 responses are valid, with zero truncations, and closed-record replay verifies the original states, questions, run, trial and phase identities, own-action histories, rewards, native requests and responses, parser, usage, and scoring. Its original post-step better-arm accuracy is 91.125%, mean reward is 58.992875, and original quadratic scoring rule is 0.896082. Mean confidence is 0.732088 and the original adjusted quadratic score is 0.919473. The original literal confidence scale remains 0 (guess) to 1 (very certain), including the source two-decimal prefix and 0.99 cap, and raw native reports are retained. These reports are not interpreted as native action probabilities or used to construct human-response NLL. Sonnet message-request round-trip P50 and P95 are 3.411 and 5.307 seconds, excluding token-count requests, and its usage-priced cost is US$9.9886 rather than an account invoice.

The remaining Claude Restless requests were canceled after Jev completed. Haiku retains 1,508 of 1,600 responses, including one invalid final format and zero truncations, with nine complete 80-trial trajectories and one incomplete trajectory. Opus retains 436 valid responses with zero truncations. All saved records pass source verification, but neither model has a complete ten-run score. No incomplete trajectory is retried or filled with generated substitutes.

### Jev and Haiku on the other tasks

Jev completes ten further human-history tasks and four autonomous tasks on their full original samples. Requests reproduce the visible information, histories, legal options, and environments of the open-model runs, and closed-record replay verifies every saved response. All 47,505 human-history and 7,537 autonomous responses are valid. Jev's two-decimal probabilities put zero mass on a human choice in 353 multiple-cue, 58 politeness, and 103 pragmatic items, so their strict NLL is infinite. Clipping at the 10^-12 of the multiple-cue scorer gives 3.069, and the social scorers define no clipping. Their usage-priced cost is US$3.97.

Jev's five largest targets (RLWM, stationary bandits, hybrid sampling, Mouselab autonomous and self-assessment) complete on their full samples in the same way.

Haiku completes temporal discounting with 7 of 9 later-reward choices. Its other completed targets are listed in [API_TIME_COST.md](API_TIME_COST.md). Its instrumental-learning run is incomplete (8 of 10 runs) and is marked part.

## Reference

Polina Tsvilodub, Kanishk Gandhi, Haoran Zhao, Jan-Philipp Fränken, Michael Franke, and Noah D. Goodman. 2025. Non-literal understanding of number words by language models. In Proceedings of the 47th Annual Conference of the Cognitive Science Society, pages 2737-2743.
