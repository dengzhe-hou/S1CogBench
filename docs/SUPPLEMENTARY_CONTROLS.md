# Supplementary controls

The current seven-model Horizon and WCST readout release is documented in [LATEST_RESULTS.md](LATEST_RESULTS.md). The original five-model P3 collection and other retained controls below keep their historical scope and counts.

This page holds the accepted records of the controlled latency study (P1), the generated reasoning conditions (P2), the common legal readouts (P3) and the non-full Claude API prefixes. These controls are separate from the Table 3 targets, and no prefix replaces a full task evaluation. Portable preparation and execution code is in [experiments/decision_controls](../experiments/decision_controls/README.md).

## Controlled latency (P1)

Controlled latency testing uses one A100 40GB device, batch size one, and 20 warm-up calls per mode before separate forward-pass and end-to-end measurements. The nine implementation configurations remain separate. Combined source acceptance covers all 54 configuration-mode cells and 37,056 timed calls. The original six successes and three authorized supplements remain source-linked, with prior failure records retained in the audit record.

P1 retains the original per-request percentile and source-identity ratio definitions. Main configurations use up to 1,000 requests per mode (576 where only 576 source requests exist). Laya TileLang, Intern Torch, and Qwen Torch retain 200 requests per mode. Twenty warm-up calls per mode are excluded from timed counts. Prior failure history remains recorded separately from the accepted supplement view. The accepted timings are in [LATENCY.md](LATENCY.md).

## Common legal readouts (P3)

A five-model comparison on the original Horizon and WCST inputs holds each model's prompt family, precision, kernel, task history, and seeds fixed while comparing legal-option argmax and sampling. Fourteen new conditions and six exact source-verified archive reuses complete the 20-condition matrix, with 474,740 executed decisions accepted against the original source, action, probability, feedback, and scoring records. Precision remains specific to model and task. Laya uses FP32 parameters with FP32 Horizon and BF16 WCST forward computation.

The P3 matrix contains 332,318 decisions from 14 newly executed conditions and 142,422 decisions from six exact archive reuses. The Horizon table is Table 4 of the paper, each of its rows covering 74,400 forced and free rows, and the WCST table is below. Both retain all five models and both readouts. No model calls are made during acceptance or table construction.


### WCST under both readouts

Each row contains 75 source participants and 12,754 executed actions. Each pair reports the participant mean followed by the trial-weighted mean, in percent.

| Model | Readout | Accuracy | Perseveration | Set loss |
| --- | --- | --- | --- | --- |
| Laya | argmax | 25.469 / 25.474 | 15.243 / 15.234 | 18.165 / 18.183 |
| Laya | sample | 25.184 / 25.153 | 16.608 / 16.583 | 17.185 / 17.179 |
| Intern | argmax | 27.896 / 27.928 | 21.246 / 21.217 | 15.840 / 15.885 |
| Intern | sample | 26.316 / 26.298 | 17.004 / 16.959 | 16.904 / 16.920 |
| Qwen | argmax | 25.196 / 25.200 | 15.879 / 15.846 | 17.881 / 17.916 |
| Qwen | sample | 25.760 / 25.772 | 16.371 / 16.395 | 17.979 / 18.002 |
| Llama | argmax | 32.367 / 32.429 | 35.494 / 35.769 | 10.710 / 10.624 |
| Llama | sample | 31.014 / 30.994 | 23.252 / 23.248 | 15.636 / 15.634 |
| Minitaur | argmax | 34.278 / 34.319 | 48.461 / 48.385 | 3.707 / 3.756 |
| Minitaur | sample | 27.073 / 27.113 | 16.909 / 16.912 | 16.826 / 16.865 |

## Generated reasoning (P2)

The original generative condition uses greedy decoding with a 1,024-token cap on Llama and Qwen. A valid final response must permit the source trajectory to advance. Invalid outputs and truncated trajectories remain recorded as incomplete. Follow-up generation is limited to at most 200 attempts each for Llama-Horizon, Qwen-Horizon, and Qwen-WCST, using a 4,096-token cap. Qwen disables native thinking in this separate condition and retains native end-of-sequence tokens, and free-text reasoning remains allowed. Its parser evaluates the final body after the last closed thinking segment and rejects an unclosed segment or multiple final-answer lines in that body. In the original Llama-WCST full shards, a matching pilot prefix is reused at most once under the original identity and history rules. The 12 proposed full follow-up shards were canceled before execution. These prefixes do not replace a full task evaluation. The original failures and the separate 25-input Qwen diagnostic with native thinking enabled retain their original scoring and output records.

### Original generation

The original P2 Llama WCST shards have a source-accepted merged rollout.

| Metric | Participant mean (%) | Trial-weighted mean (%) |
| --- | ---: | ---: |
| Accuracy | 25.923 | 25.906 |
| Perseveration error | 19.428 | 19.437 |
| Set-loss error | 14.815 | 14.835 |

This is original Llama WCST generation with the 1,024-token cap. Acceptance covers 75 participants and 12,754 saved actions, including 200 exact pilot calls reused once and 12,554 new calls. The outputs include 2,217 invalid and 2,216 truncated responses, retained in the original trajectory and scoring records. Scores follow the original executed actions, and completeness does not imply all responses are legal. This condition is separate from the 4,096-token follow-up prefixes.

Original Llama Horizon and Qwen Horizon and WCST remain incomplete, so the original P2 batch is not reported as fully completed. The separate thinking-enabled Qwen diagnostic verifies all 25 saved inputs and responses, with zero legal outputs under its original strict parser. Of these, 24 reach the 4,096-token cap, and one Horizon output closes thinking and emits native EOS but is rejected because the original strict parser sees multiple final-answer lines. Its original scores remain unchanged. The 12 canceled full follow-up shards remain unexecuted, and the retained Qwen WCST follow-up remains at 130 of 200 attempts.

### Retained local generation prefixes

The Llama-Horizon and Qwen-Horizon follow-up conditions each reached 200 attempts. Qwen-WCST instead closed after 130 attempts: 55 produced a legal final response, and 75 reached the 4,096-token cap. Each of the 75 selected participant trajectories eventually stopped at a capped output, so no trajectory completed and the 200-call gate was not reached. These incomplete records are retained without retry. CPU source replay verifies the saved state, native or raw prompt, identity, history, action, feedback, token-cap and parser outcomes, and timing statistics. Record integrity does not convert an incomplete pilot into a completed evaluation, and the canceled full follow-up shards remain unexecuted.

| Condition | Attempts | Valid | Invalid | Capped | P50 (s) | P95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-Horizon | 200/200 | 197 | 3 | 3 | 2.958 | 4.650 |
| Qwen-Horizon | 200/200 | 200 | 0 | 0 | 16.330 | 50.001 |
| Qwen-WCST | 130/200 | 55 | 75 | 75 | 146.453 | 454.076 |

These prefixes use a 4,096-token cap. Attempts are shown against the planned 200-call pilot. Valid denotes a legal final response, not a correct cognitive decision. Qwen disables native thinking in this condition. Qwen-WCST ends with an incomplete pilot, and its 130 saved attempts do not stand in for 200 calls or a full task evaluation. Each model retains its source prompt family and action-dependent history. Timing follows each saved batch-one synchronized local generation measurement and is separate from the API round-trip timer below. Original 1,024-token results and the separate thinking-enabled Qwen diagnostic retain their own records and scoring.

## Claude API decision prefixes

Separate API conditions use `claude-opus-5-5`, `claude-sonnet-5-5`, and `claude-haiku-5-5` on at most 200 source decisions per model and task for Horizon and WCST. Each explicitly sets native adaptive thinking, medium effort, summarized display, and a 4,096-token output cap. The earlier Opus run has a US$40 ceiling, and the additional Sonnet and Haiku runs share a US$100 ceiling. Medium is an explicit setting, not a guarantee of equal internal computation. Sonnet's documented native default is high. Native thinking blocks and answer text are saved separately, and the strict final-answer parser reads only native text blocks. Source inputs, environment seeds, histories, and feedback are retained.

The API does not expose a generation seed, numerical precision, or legal-action probabilities, so this condition is not treated as a matched greedy and BF16 or human-response NLL comparison. Its non-streaming timer measures the complete model-request round trip, including network and provider service time, but excludes the preceding token-count request and local audit-file writes. Time to first token and provider-only forward and decode time are not measured. Returned usage supports token and cost accounting, and usage-priced totals are not an account invoice. No retry or full-task continuation is performed.

### Retained API decision prefixes

Opus 5.5 retains 200 Horizon and 200 WCST attempts, with 400 legal final responses, 0 invalid responses, and 0 capped outputs. Sonnet 5.5 retains 200 Horizon and 200 WCST attempts, with 400 legal final responses, 0 invalid responses, and 0 capped outputs. Haiku 5.5 retains 200 Horizon and 200 WCST attempts, with 395 legal final responses, 5 invalid responses, and 5 capped outputs. Closed-record acceptance replays every original state, question, response identity, history, executed choice, and source feedback, and reconciles native request, response, and usage records. These checks establish prefix-record and interface validity. Full-task performance and human-response NLL remain unmeasured for these API conditions.

Usage-priced totals are US$6.2724 for Opus 5.5, US$2.8402 for Sonnet 5.5, and US$0.2077 for Haiku 5.5. They are calculated from returned token usage and are not account invoices. Models retain their own action-dependent histories and actual input lengths. No failed or capped call is retried, and no full-task continuation is performed.

| Model | Task | Calls | Valid | Capped | P50 (s) | P95 (s) | Input mean | Output mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Opus 5.5 | Horizon | 200 | 200 | 0 | 3.238 | 5.530 | 459.3 | 234.9 |
| Opus 5.5 | WCST | 200 | 200 | 0 | 5.309 | 11.336 | 3757.5 | 489.9 |
| Sonnet 5.5 | Horizon | 200 | 200 | 0 | 2.729 | 3.784 | 459.3 | 202.1 |
| Sonnet 5.5 | WCST | 200 | 200 | 0 | 3.654 | 6.758 | 3759.9 | 374.2 |
| Haiku 5.5 | Horizon | 200 | 200 | 0 | 3.262 | 4.665 | 459.3 | 365.9 |
| Haiku 5.5 | WCST | 200 | 195 | 5 | 6.433 | 17.362 | 1348.4 | 1349.9 |

Valid denotes a legal final response under the fixed parser, not a correct cognitive decision. Capped counts output truncation, and invalid responses and their request times remain in the records. Input and output columns are mean provider-reported token usage per request, and output includes native thinking. Timing is non-streaming model-request round trip. Each condition permits at most 200 attempts and is not a full Horizon or WCST evaluation. These records do not provide legal-action probabilities or human-response NLL.

### Task performance over saved API prefixes

This table scores the existing 200 attempts per model and task, with no further API requests or environment rollouts.

| Task mode | Prefix target | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 |
| --- | --- | ---: | ---: | ---: |
| Horizon | Mean reward | 56.8700 | 56.9450 | 56.8550 |
| WCST | Accuracy (%, correct/200) | 62.5 (125/200) | 65.0 (130/200) | 66.5 (133/200) |

Horizon routes each saved free choice to the original timeline's reward for that arm and averages the 200 rewards. WCST uses the original executed-card accuracy, retaining invalid or capped outputs as incorrect and keeping all 200 attempts in the denominator. All three Horizon prefixes cover the same 200 source identities from one timeline. Opus and Sonnet's WCST prefixes each cover two source subjects. Haiku's five capped outputs terminate those subject trajectories, and its prefix reaches six subjects. These different source compositions and model-dependent histories are retained. The panel provides observed prefix performance, separate from the full 34,720-decision Horizon and 12,754-decision WCST evaluations. It does not establish a full-task model ranking or an independent-population confidence interval. These are saved-prefix measurements, not full-task performance estimates.

### Common-source comparison of saved decisions

This table intersects the accepted source identities across all three Claude models and both common-readout conditions of all five local models. It retains 200 Horizon free choices from source timeline 1 and 74 WCST decisions from source subjects 1 and 3 (32 and 42 decisions). Horizon identity comprises timeline, block, game, and source trial. WCST identity comprises subject, trial, and source step. Matching those identities does not make inputs identical: models retain their own executed actions and feedback histories, prompt families, precision, and inference settings. Original source rewards, correct-card labels, invalid-response handling, and scoring are unchanged. Invalid WCST attempts remain in the denominator. No inference, API request, environment rollout, or refitting is added by this analysis.

| Model | Readout | Horizon reward | WCST correct | WCST (%) | Invalid WCST |
| --- | --- | ---: | ---: | ---: | ---: |
| Opus 5.5 | native FINAL | 56.8700 | 47/74 | 63.51 | 0 |
| Sonnet 5.5 | native FINAL | 56.9450 | 49/74 | 66.22 | 0 |
| Haiku 5.5 | native FINAL | 56.8550 | 49/74 | 66.22 | 2 |
| Laya | argmax | 50.2300 | 15/74 | 20.27 | 0 |
| Laya | sample | 50.0500 | 17/74 | 22.97 | 0 |
| Intern-Decision-4B | argmax | 56.2150 | 25/74 | 33.78 | 0 |
| Intern-Decision-4B | sample | 51.3450 | 22/74 | 29.73 | 0 |
| Llama3.1-8B | argmax | 54.3250 | 28/74 | 37.84 | 0 |
| Llama3.1-8B | sample | 51.5650 | 29/74 | 39.19 | 0 |
| Minitaur8B | argmax | 56.4750 | 32/74 | 43.24 | 0 |
| Minitaur8B | sample | 52.9200 | 20/74 | 27.03 | 0 |
| Qwen3.5-4B | argmax | 52.6750 | 14/74 | 18.92 | 0 |
| Qwen3.5-4B | sample | 51.3300 | 17/74 | 22.97 | 0 |

Horizon uses the same 200 free-choice identities from one source timeline, and WCST uses 74 identities from two source participants. Every row retains its own action-dependent history. Both local legal readouts are shown. Haiku's two invalid WCST attempts count as incorrect. These scores are separate from the 200-attempt scores in the previous table, and the subset does not add independent participants or establish a full-task ranking.

On the 74 common WCST decisions, Opus, Sonnet, and Haiku score 47/74, 49/74, and 49/74. The original 200-attempt scores remain 125/200, 130/200, and 133/200. Haiku has two invalid attempts in the common subset and five in its original prefix. The Horizon subset is unchanged at 200 decisions, yielding mean rewards 56.8700, 56.9450, and 56.8550. This records the observed source-matched subset without a population confidence interval from trajectory steps.
