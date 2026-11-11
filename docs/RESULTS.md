# Accepted results

The current release covers 24 tasks and seven open models, with all 168 task and model pairs complete. It supplies 189 primary scores and 70 Horizon and WCST readout metrics. See [LATEST_RESULTS.md](LATEST_RESULTS.md) for the source and version details. The original five-model arXiv snapshot remains available separately. The 31-task catalogue retains seven deferred tasks. Existing API rows retain Jev 1.14 in 27 modes and Claude 5.5 on selected modes. A table cell printed as n.a., part., n.e., n.f., canc. or inv. retains its marker and reason. See [API_EXTENSION.md](API_EXTENSION.md) and [API_TIME_COST.md](API_TIME_COST.md).

Download current JSON/CSV from results/latest/ and the historical arXiv files from results/aggregate/. Keep task, protocol, mode, metric variant, scope, target, unit and aggregation together. Report denominator n. Infinity remains an infinite NLL, and undefined values are not zero. Fine-grained execution logs remain under their source access terms. Original and compact presentations, human prediction and autonomous execution, and binary-event versus class-summed Brier remain distinct.

The compact paper appendix keeps the central comparisons; the complete aggregate table and task catalogue retain all accepted condition results, including negative results. See DATA.md, PROTOCOLS.md and CPU_REUSE_VALIDATION_24.md for access and verification boundaries.

## Uncertainty and retained evidence

Horizon condition contrasts use six held-out participants, and their intervals are pointwise and unadjusted. RLWM load pairs all 79 participants at set sizes 6 and 2, while presentation-1/10 learning contrasts at size 6 pair 69. Its intervals use 5,000 participant bootstrap resamples. Full curves retain every source-defined set size and observed presentation. Trial counts are not independent participant replications.

The repository provides the full condition tables, both grammar encodings (below), task-specific metric applicability, complete aggregate results, and extended protocol notes. Negative values, infinite NLL, and the undefined all-invalid MAE are retained.

## Grammar response likelihoods

This diagnostic table retains the original five-model snapshot. It is separate from the current seven-model primary-score table.

| Model | Binned NLL | Zero targets | Integer/choice NLL | Zero targets |
| --- | ---: | ---: | ---: | ---: |
| Laya | +∞ | 5 | +∞ | 28 |
| Intern-Decision-4B | 1.7602 | 0 | 1.9535 | 0 |
| Llama3.1-8B | 2.0106 | 0 | 2.2362 | 0 |
| Minitaur8B | 1.4391 | 0 | 1.6204 | 0 |
| Qwen3.5-4B | 2.0549 | 0 | 2.2495 | 0 |
| Lookup (official) | 1.4127 | 0 | -- | -- |
| Uniform over 11 source bins | 2.3979 | 0 | -- | -- |

These use the same 323 held-out participants and all 9,044 original responses (28 per participant). Count reports use floor(value/2), percentile reports floor(value/10), and difficulty reports their original integer. Answer labels follow the original per-question category order. Neural binned probabilities sum all original legal-integer probabilities, while full-integer/choice NLL is retained. Zero targets count saved target probabilities equal to zero, and no response is dropped or clipped.

Lookup uses the original 1,000-iteration fit on 3,192 training participants and 89,376 responses, and its author-archived binned NLL is 1.4126839638. It retains the 11-class softmax denominator, including extra answer classes. Neural typed legal-support distributions and training histories differ, so this is an adapted same-target comparison. Lookup supplies no within-bin integer distribution (shown as --), and no logic OOD baseline is constructed.

## Minitaur on nonliteral number

Minitaur's MAE on nonliteral number is 28.53. Versions before 2026-10-31 marked all 330 reports invalid, which was a parser error on our side. All 330 saved completions start with a number and the closing marker, for example `50.0%».`, but generation ran on for its 16 tokens and the parser rejected the trailing text. `parse_percentage` in `src/social_numeric_models.py` now reads the number before the marker, and [`results/aggregate/minitaur_nonliteral_reparsed.csv`](../results/aggregate/minitaur_nonliteral_reparsed.csv) holds the re-read reports. No model was called again. Minitaur reports 50% on 319 of the 330 items, so its 28.53 sits close to the 27.20 of always reporting 50%. The other models had no invalid reports on this task.

## Details the paper does not print

These details left the paper's appendices on 2026-10-31.

- A participant holdout differs from learning and test phases within that participant.
- Autonomous Horizon timelines share a source forced-observation schedule, so they are not 31 independently sampled initial-information schedules.
- RLWM curves retain all set sizes 2 to 6 and presentations 1 to 15, including late responses. [rlwm_curves.csv](../results/aggregate/rlwm_curves.csv) holds every point for the five open models, the human participants and the seven archived fits, with pointwise 95% Student-t participant intervals.
- The compact multiple-cue protocol retains all responses, their order, and visible feedback.
- Across the Horizon and WCST readouts, each model keeps its own prompt, precision, kernel, history protocol, and seeds.
- WCST perseveration is a wrong choice that repeats the rule of a wrong previous choice, and set loss is a wrong choice that leaves the rule of a correct one. Both are rates over all trials. For uniform choice each is 17.13%, an exact expectation.
- Paradigms are deduplicated across collections, so a new seed, prompt, condition, or transcription does not create a new task.

The paper's cognitive profiles figure follows these conventions.

- Panel a uses 960 first free choices per model, with the observed human choice contrast as its reference, and is unadjusted.
- In panel b the weaker parameter takes values 0.4, 0.5, or 0.6 and is not always uninformative, each model answers 400 questions, and Bayes MAE is zero.
- In panel c, reversals are averaged within each run before the means. Offsets 0 to 18 include three reversals per run and offset 19 includes two. Minitaur's flat row reflects a fixed action and the sampled environment orientations, and its change at offset 19 comes from the two remaining reversals. Minitaur chooses F on all 800 trials across ten runs, although its probability of F varies from 0.5622 to 0.9325. That action scores 63.33% in the three later reward blocks and 49.00% overall.

## Table 3

[table2.csv](../results/aggregate/table2.csv) is Table 3 together with the Opus and Sonnet columns, which the paper prints in a sentence of its appendix, and [table2.json](../results/aggregate/table2.json) names the file and row behind each of its 297 cells. `python scripts/build_table2.py` rebuilds both from `results.json` and `reference_results.json` alone.

A row of Table 3 can draw on different protocols, because each model keeps the readout it was accepted under. Multiple-cue judgment uses the full prefix for Llama, Minitaur and Jev and the compact history for the others. In Horizon autonomous, Stationary bandits and Mouselab autonomous, Laya, Llama and Minitaur have their own row family. Use `table2.json` to find the row behind a cell.

| Mark | Meaning |
| --- | --- |
| n.e. | Outside the measured full-task scope |
| n.a. | Lack a full probability score |
| n.f. | Only non-full prefixes exist |
| part. | Incomplete records are retained |
| canc. | Canceled before generation |
| s after a Haiku value | Haiku stated option probabilities, a different readout from the open models |
| † | Recovered fit or policy |
| § | Same-participant fit |
| ‡ | Within-participant holdout |
| ∗ | New autonomous policy reference |
| ¶ | Reference added for the remaining rows |

The reference type names fits to participants' responses (human fit), other participants' responses without a fitted model (human), the analytic solution or answer key (optimal), and policies without human fitting (algorithm). The domain is only an organizational label. The paper's bold and gray apply to the five open models, gray marking scores not better than Uniform, and the CSV does not carry them. [BASELINES.md](BASELINES.md) defines the Uniform values and the references.

Temporal discounting reports the later-reward rate over realized reward questions, a preference rather than a score. The open-model rows are `t00001` to `t00005` (0/6, 1/6, 0/6, 8/10 and 9/10 later-reward choices for Laya, Intern, Llama, Minitaur and Qwen), computed from saved choices with no new model call.
